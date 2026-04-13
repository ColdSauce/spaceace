"""PPO training environment — composes strategies, delegates all math to them.

Historically this file was ~300 lines containing observation augmentation,
reward shaping, pathfinder queries, and metric tracking inline. All of that is
now in `spaceace.strategies`. This file is the glue that wires a
SpaceAceGymWrapper together with the chosen strategies and exposes SB3's
`make_env` / `MetricsCallback` helpers unchanged for backward compat.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from spaceace.core.gym_wrapper import SpaceAceGymWrapper
from spaceace.strategies import (
    DenseShapedReward,
    PathAugmentedObs23,
    RustPathfinder,
)


class SpaceAceTrainingEnv(gym.Wrapper):
    """Glue wrapper: applies ObservationBuilder + RewardShaper and supports action repeat."""

    def __init__(
        self,
        env: gym.Env,
        level: int = 0,
        max_steps: int = 3000,
        action_repeat: int = 5,
    ):
        super().__init__(env)
        self._max_steps = max_steps
        self._action_repeat = action_repeat

        pathfinder = RustPathfinder(level)
        self._obs_builder = PathAugmentedObs23(pathfinder, max_steps)
        self._reward_shaper = DenseShapedReward(pathfinder, max_steps)
        self.observation_space = self._obs_builder.space

        self._last_metrics: dict = {}

    @property
    def last_episode_completed(self) -> bool:
        return bool(self._last_metrics.get("completed", False))

    @property
    def last_episode_crashed(self) -> bool:
        return bool(self._last_metrics.get("crashed", False))

    @property
    def last_episode_steps(self) -> int:
        return int(self._last_metrics.get("length", 0))

    @property
    def last_episode_pickups_collected(self) -> int:
        return int(self._last_metrics.get("pickups_collected", 0))

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)
        self._reward_shaper.reset(obs, info, self.env)
        return self._obs_builder.reset(obs, info, self.env), info

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        action_arr = np.asarray(action)

        for _ in range(self._action_repeat):
            obs, _base_reward, terminated, truncated, info = self.env.step(action)
            total_reward += self._reward_shaper.shape(obs, action_arr, info, self.env)
            if terminated or truncated:
                break

        if terminated or truncated:
            metrics = self._reward_shaper.episode_metrics()
            self._last_metrics = metrics
            info["episode_metrics"] = metrics

        return (
            self._obs_builder.build(obs, info, self.env),
            total_reward,
            terminated,
            truncated,
            info,
        )


class MetricsCallback(BaseCallback):
    """Logs per-episode metrics from info['episode_metrics'] to TensorBoard."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            metrics = info.get("episode_metrics")
            if metrics is not None:
                self.logger.record("episode/thrust_ratio", metrics["thrust_ratio"])
                self.logger.record("episode/pickups_collected", metrics["pickups_collected"])
                self.logger.record("episode/crashed", float(metrics["crashed"]))
                self.logger.record("episode/completed", float(metrics["completed"]))
                self.logger.record("episode/length", metrics["length"])
        return True


def make_env(level: int, max_steps: int, action_repeat: int = 5):
    """SB3-style env factory: SpaceAceGymWrapper -> strategies -> Monitor."""

    def _init():
        base = SpaceAceGymWrapper(level=level, max_steps=max_steps)
        shaped = SpaceAceTrainingEnv(
            base, level=level, max_steps=max_steps, action_repeat=action_repeat
        )
        return Monitor(shaped)

    return _init
