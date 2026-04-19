"""Vec-env factory composing strategies into a gymnasium Env usable by any trainer."""

from __future__ import annotations

import os

# Keep BLAS single-threaded inside each SubprocVecEnv worker; otherwise the
# spawned workers oversubscribe the CPU and tank FPS on macOS (Accelerate).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from collections import OrderedDict
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from spaceace.core.gym_wrapper import SpaceAceGymWrapper
from spaceace.strategies import (
    ObservationBuilder,
    RewardShaper,
    RustPathfinder,
    resolve,
)
from spaceace.strategies.actions import ALL_ACTIONS


class StrategyWrapper(gym.Wrapper):
    """Binds an ObservationBuilder + RewardShaper to a SpaceAceGymWrapper.

    Exposes Discrete(6) to the agent and decodes to the underlying
    MultiDiscrete([2,2,2]) via ALL_ACTIONS. Supports action repeat
    (hold action for N physics frames).
    """

    def __init__(
        self,
        env: gym.Env,
        observation: ObservationBuilder,
        reward: RewardShaper,
        action_repeat: int = 1,
        pathfinder=None,
        gamma: float = 0.995,
    ):
        super().__init__(env)
        self._obs = observation
        self._reward = reward
        self._action_repeat = action_repeat
        self._pathfinder = pathfinder
        self._gamma = gamma
        self.observation_space = observation.space
        self.action_space = gym.spaces.Discrete(len(ALL_ACTIONS))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._pathfinder is not None and hasattr(self._pathfinder, 'reset_sticky'):
            self._pathfinder.reset_sticky()
        self._reward.reset(obs, info, self.env)
        return self._obs.reset(obs, info, self.env), info

    def step(self, action):
        # Accept either a Discrete int (from PPO) or a raw MultiDiscrete
        # triplet (from MCTS kickstart, which produces ALL_ACTIONS entries).
        arr = np.asarray(action)
        if arr.shape == () or arr.shape == (1,):
            decoded = ALL_ACTIONS[int(arr)]
        else:
            decoded = arr.astype(np.int32, copy=False)

        total_reward = 0.0
        discount = 1.0
        terminated = False
        truncated = False
        pf = self._pathfinder
        for _ in range(self._action_repeat):
            if pf is not None:
                pf.clear_cache()
            obs, base_reward, terminated, truncated, info = self.env.step(decoded)
            info["_base_reward"] = base_reward
            total_reward += discount * self._reward.shape(obs, decoded, info, self.env)
            discount *= self._gamma
            if terminated or truncated:
                break
        if terminated or truncated:
            info["episode_metrics"] = self._reward.episode_metrics()
        return self._obs.build(obs, info, self.env), total_reward, terminated, truncated, info


def _build_strategies(
    level: int,
    max_steps: int,
    obs_key: str,
    reward_key: str,
    pathfinder_backend: str = "grid",
):
    pathfinder = RustPathfinder(level, backend=pathfinder_backend)
    ObsCls = resolve("observation", obs_key)
    RewardCls = resolve("reward", reward_key)
    # Classes that need pathfinder/max_steps accept them; simple ones don't.
    import inspect

    obs_params = inspect.signature(ObsCls.__init__).parameters
    if "pathfinder" in obs_params:
        obs = ObsCls(pathfinder, max_steps)
    else:
        obs = ObsCls()

    reward_params = inspect.signature(RewardCls.__init__).parameters
    if "pathfinder" in reward_params:
        reward = RewardCls(pathfinder, max_steps)
    else:
        reward = RewardCls()

    return obs, reward, pathfinder


class RandomLevelEnv(gym.Wrapper):
    """On each reset, pick a random level from a pool.

    Used by curriculum training so a single VecEnv samples across all levels
    in the current stage.
    """

    # Keep at most this many per-level StrategyWrappers alive per worker.
    # Pathfinder construction runs one Dijkstra per pickup (slow), so we avoid
    # rebuilding on every reset. Curriculum stages usually pool 1-10 levels,
    # so 16 is generous without blowing memory (roughly a few MB/level).
    _CACHE_SIZE = 16

    def __init__(
        self,
        levels: list[int],
        max_steps: int,
        obs_key: str = "path_augmented",
        reward_key: str = "dense_shaped",
        action_repeat: int = 5,
        pathfinder_backend: str = "grid",
    ):
        self._levels = levels
        self._max_steps = max_steps
        self._obs_key = obs_key
        self._reward_key = reward_key
        self._action_repeat = action_repeat
        self._pathfinder_backend = pathfinder_backend
        self._wrapper_cache: OrderedDict[int, StrategyWrapper] = OrderedDict()
        wrapped = self._wrapper_for(levels[0])
        super().__init__(wrapped)

    def _wrapper_for(self, level: int) -> StrategyWrapper:
        # Env var escape hatch for benchmarking the old "rebuild-every-reset" path.
        if os.environ.get("SPACEACE_DISABLE_WRAPPER_CACHE") != "1":
            wrapper = self._wrapper_cache.get(level)
            if wrapper is not None:
                self._wrapper_cache.move_to_end(level)
                return wrapper
        base = SpaceAceGymWrapper(level=level, max_steps=self._max_steps)
        obs_strategy, reward_strategy, pf = _build_strategies(
            level, self._max_steps, self._obs_key, self._reward_key, self._pathfinder_backend,
        )
        wrapper = StrategyWrapper(
            base, obs_strategy, reward_strategy,
            action_repeat=self._action_repeat, pathfinder=pf,
        )
        self._wrapper_cache[level] = wrapper
        if len(self._wrapper_cache) > self._CACHE_SIZE:
            self._wrapper_cache.popitem(last=False)
        return wrapper

    def set_curriculum(self, levels: list[int], max_steps: int):
        """Update the level pool and max_steps. Takes effect on next reset."""
        self._levels = levels
        if max_steps != self._max_steps:
            # max_steps baked into SpaceAceGymWrapper + strategies; invalidate cache.
            self._max_steps = max_steps
            self._wrapper_cache.clear()

    def reset(self, **kwargs):
        import random

        level = random.choice(self._levels)
        self.env = self._wrapper_for(level)
        return self.env.reset(**kwargs)


class _CurriculumMonitor(Monitor):
    """Monitor that forwards set_curriculum to the inner RandomLevelEnv.

    Needed so SubprocVecEnv.env_method("set_curriculum", ...) works through
    the Monitor wrapper.
    """

    def set_curriculum(self, levels: list[int], max_steps: int):
        self.env.set_curriculum(levels, max_steps)


def make_random_level_env(
    levels: list[int],
    max_steps: int,
    obs: str = "path_augmented",
    reward: str = "dense_shaped",
    action_repeat: int = 5,
    pathfinder_backend: str = "grid",
) -> Callable[[], gym.Env]:
    """Thunk that builds a RandomLevelEnv wrapped in Monitor."""

    def _init() -> gym.Env:
        return _CurriculumMonitor(RandomLevelEnv(levels, max_steps, obs, reward, action_repeat, pathfinder_backend))

    return _init


def make_single_env(
    level: int,
    max_steps: int,
    obs: str = "path_augmented",
    reward: str = "dense_shaped",
    action_repeat: int = 5,
    pathfinder_backend: str = "grid",
) -> Callable[[], gym.Env]:
    """Return a no-arg thunk that builds one wrapped, monitored env.

    Thunks are what SB3 VecEnvs expect. Strategies are built inside the thunk
    so each subprocess worker owns its own instances.
    """

    def _init() -> gym.Env:
        base = SpaceAceGymWrapper(level=level, max_steps=max_steps)
        obs_strategy, reward_strategy, pf = _build_strategies(level, max_steps, obs, reward, pathfinder_backend)
        wrapped = StrategyWrapper(base, obs_strategy, reward_strategy, action_repeat=action_repeat, pathfinder=pf)
        return Monitor(wrapped)

    return _init


def make_vec_env(
    level: int,
    max_steps: int,
    n_envs: int,
    obs: str = "path_augmented",
    reward: str = "dense_shaped",
    action_repeat: int = 5,
    subprocess: bool = True,
    pathfinder_backend: str = "grid",
) -> VecEnv:
    """One call, every agent. Use subprocess=False for eval envs."""
    thunks = [
        make_single_env(level, max_steps, obs, reward, action_repeat, pathfinder_backend)
        for _ in range(n_envs)
    ]
    # DummyVecEnv default (see note in sb3_trainer._make_curriculum_vec_env).
    if os.environ.get("SPACEACE_USE_SUBPROC") == "1" and subprocess and n_envs > 1:
        return SubprocVecEnv(thunks, start_method="spawn")
    return DummyVecEnv(thunks)
