"""Vec-env factory composing strategies into a gymnasium Env usable by any trainer."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from spaceace.core.gym_wrapper import SpaceAceGymWrapper
from spaceace.strategies import (
    DenseShapedReward,
    ObservationBuilder,
    PathAugmentedObs23,
    RawObs19,
    RewardShaper,
    RustPathfinder,
    SparseReward,
)


class StrategyWrapper(gym.Wrapper):
    """Binds an ObservationBuilder + RewardShaper to a SpaceAceGymWrapper.

    Supports action repeat (hold action for N physics frames). Does not touch
    the action space — agents are expected to already speak the underlying
    MultiDiscrete([2,2,2]) encoding.
    """

    def __init__(
        self,
        env: gym.Env,
        observation: ObservationBuilder,
        reward: RewardShaper,
        action_repeat: int = 1,
    ):
        super().__init__(env)
        self._obs = observation
        self._reward = reward
        self._action_repeat = action_repeat
        self.observation_space = observation.space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reward.reset(obs, info, self.env)
        return self._obs.reset(obs, info, self.env), info

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        action_arr = np.asarray(action)
        for _ in range(self._action_repeat):
            obs, base_reward, terminated, truncated, info = self.env.step(action)
            info["_base_reward"] = base_reward
            total_reward += self._reward.shape(obs, action_arr, info, self.env)
            if terminated or truncated:
                break
        if terminated or truncated:
            info["episode_metrics"] = self._reward.episode_metrics()
        return self._obs.build(obs, info, self.env), total_reward, terminated, truncated, info


def _build_strategies(level: int, max_steps: int, obs_key: str, reward_key: str):
    pathfinder = RustPathfinder(level)
    if obs_key == "raw":
        obs = RawObs19()
    elif obs_key == "path_augmented":
        obs = PathAugmentedObs23(pathfinder, max_steps)
    else:
        raise ValueError(f"unknown observation strategy: {obs_key}")
    if reward_key == "sparse":
        reward = SparseReward()
    elif reward_key == "dense_shaped":
        reward = DenseShapedReward(pathfinder, max_steps)
    else:
        raise ValueError(f"unknown reward strategy: {reward_key}")
    return obs, reward


class RandomLevelEnv(gym.Wrapper):
    """On each reset, pick a random level from a pool.

    Used by curriculum training so a single VecEnv samples across all levels
    in the current stage.
    """

    def __init__(
        self,
        levels: list[int],
        max_steps: int,
        obs_key: str = "path_augmented",
        reward_key: str = "dense_shaped",
        action_repeat: int = 5,
    ):
        self._levels = levels
        self._max_steps = max_steps
        self._obs_key = obs_key
        self._reward_key = reward_key
        self._action_repeat = action_repeat
        initial_level = levels[0]
        base = SpaceAceGymWrapper(level=initial_level, max_steps=max_steps)
        obs_strategy, reward_strategy = _build_strategies(initial_level, max_steps, obs_key, reward_key)
        wrapped = StrategyWrapper(base, obs_strategy, reward_strategy, action_repeat=action_repeat)
        super().__init__(wrapped)

    def reset(self, **kwargs):
        import random

        level = random.choice(self._levels)
        base = SpaceAceGymWrapper(level=level, max_steps=self._max_steps)
        obs_strategy, reward_strategy = _build_strategies(
            level, self._max_steps, self._obs_key, self._reward_key
        )
        self.env = StrategyWrapper(base, obs_strategy, reward_strategy, action_repeat=self._action_repeat)
        return self.env.reset(**kwargs)


def make_random_level_env(
    levels: list[int],
    max_steps: int,
    obs: str = "path_augmented",
    reward: str = "dense_shaped",
    action_repeat: int = 5,
) -> Callable[[], gym.Env]:
    """Thunk that builds a RandomLevelEnv wrapped in Monitor."""

    def _init() -> gym.Env:
        return Monitor(RandomLevelEnv(levels, max_steps, obs, reward, action_repeat))

    return _init


def make_single_env(
    level: int,
    max_steps: int,
    obs: str = "path_augmented",
    reward: str = "dense_shaped",
    action_repeat: int = 5,
) -> Callable[[], gym.Env]:
    """Return a no-arg thunk that builds one wrapped, monitored env.

    Thunks are what SB3 VecEnvs expect. Strategies are built inside the thunk
    so each subprocess worker owns its own instances.
    """

    def _init() -> gym.Env:
        base = SpaceAceGymWrapper(level=level, max_steps=max_steps)
        obs_strategy, reward_strategy = _build_strategies(level, max_steps, obs, reward)
        wrapped = StrategyWrapper(base, obs_strategy, reward_strategy, action_repeat=action_repeat)
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
) -> VecEnv:
    """One call, every agent. Use subprocess=False for eval envs."""
    thunks = [
        make_single_env(level, max_steps, obs, reward, action_repeat)
        for _ in range(n_envs)
    ]
    if subprocess and n_envs > 1:
        return SubprocVecEnv(thunks, start_method="fork")
    return DummyVecEnv(thunks)
