"""Reward shapers. Sparse passthrough for MCTS/AlphaZero, dense for PPO."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from spaceace.strategies.base import Pathfinder, RewardShaper


class SparseReward(RewardShaper):
    """Passes through the base Rust-computed reward unchanged."""

    def reset(self, raw_obs: np.ndarray, info: dict, env) -> None:
        pass

    def shape(self, raw_obs: np.ndarray, action: np.ndarray, info: dict, env) -> float:
        return float(info.get("_base_reward", 0.0))


class DenseShapedReward(RewardShaper):
    """Minimal shaped reward: pickup good, crash bad, path delta for gradient.

    Only three signals:
    1. Pickup collected: big positive
    2. Crash: big negative
    3. Path distance delta: small shaping to guide toward pickups

    Everything else (speed, TTI, overspeed) is left for the agent to discover
    through the crash penalty.
    """

    STEP_COST = -0.01
    CRASH_PENALTY = -200.0
    LEVEL_COMPLETE_BONUS = 1000.0
    PICKUP_BONUS = 100.0
    PATH_DIST_DELTA_SCALE = 0.2

    def __init__(self, pathfinder: Pathfinder, max_steps: int) -> None:
        self._pathfinder = pathfinder
        self._max_steps = max_steps
        self._prev_pickups_remaining: int | None = None
        self._prev_path_dist: float | None = None
        self._steps = 0
        self._thrust_steps = 0
        self._pickups_collected = 0
        self._crashed = False
        self._completed = False

    def reset(self, raw_obs: np.ndarray, info: dict, env) -> None:
        self._prev_pickups_remaining = int(raw_obs[16])
        collected = list(env.get_pickup_states())
        path_dist, _, _ = self._pathfinder.nearest_pickup_info(
            float(raw_obs[0]), float(raw_obs[1]), collected
        )
        self._prev_path_dist = path_dist
        self._steps = 0
        self._thrust_steps = 0
        self._pickups_collected = 0
        self._crashed = False
        self._completed = False

    def shape(self, obs: np.ndarray, action: np.ndarray, info: dict, env) -> float:
        self._steps += 1
        if int(action[2]) > 0:
            self._thrust_steps += 1

        pickups_now = info.get("pickups_remaining", self._prev_pickups_remaining or 0)
        collected = 0
        if self._prev_pickups_remaining is not None:
            collected = self._prev_pickups_remaining - pickups_now
            if collected > 0:
                self._pickups_collected += collected
        self._prev_pickups_remaining = pickups_now

        if info.get("ship_exploded", False):
            self._crashed = True
            return self.CRASH_PENALTY
        if info.get("level_completed", False):
            self._completed = True
            time_remaining = 1.0 - self._steps / self._max_steps
            return self.LEVEL_COMPLETE_BONUS + 500.0 * time_remaining

        reward = self.STEP_COST

        if collected > 0:
            reward += collected * self.PICKUP_BONUS

        # Path distance delta — guide toward nearest pickup.
        if pickups_now >= 1:
            pickup_states = list(env.get_pickup_states())
            path_dist, _, _ = self._pathfinder.nearest_pickup_info(
                float(obs[0]), float(obs[1]), pickup_states
            )

            if collected > 0:
                self._prev_path_dist = path_dist
            else:
                if self._prev_path_dist is not None:
                    reward += (self._prev_path_dist - path_dist) * self.PATH_DIST_DELTA_SCALE
                self._prev_path_dist = path_dist

        return reward

    def episode_metrics(self) -> dict[str, Any]:
        total = max(self._steps, 1)
        return {
            "thrust_ratio": self._thrust_steps / total,
            "pickups_collected": self._pickups_collected,
            "crashed": self._crashed,
            "completed": self._completed,
            "length": self._steps,
        }
