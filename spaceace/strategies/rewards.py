"""Reward shapers. Sparse passthrough for MCTS/AlphaZero, dense for PPO."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from spaceace.strategies.base import Pathfinder, RewardShaper
from spaceace.strategies.observation import compute_min_tti


class SparseReward(RewardShaper):
    """Passes through the base Rust-computed reward unchanged.

    Signature differs from DenseShapedReward: it receives `base_reward` from the
    wrapper and returns it. The wrapper handles that via `shape_sparse()` instead
    of `shape()` — see StrategyWrapper.
    """

    def reset(self, raw_obs: np.ndarray, info: dict, env) -> None:
        pass

    def shape(self, raw_obs: np.ndarray, action: np.ndarray, info: dict, env) -> float:
        return float(info.get("_base_reward", 0.0))


class DenseShapedReward(RewardShaper):
    """PPO's hand-tuned shaped reward: path-delta, proximity, velocity, TTI, overspeed.

    Every constant is frozen at the value used to train the PPO checkpoints so
    observation/reward math stays byte-identical for fresh re-training.
    """

    STEP_COST = -0.01
    CRASH_PENALTY = -50.0
    LEVEL_COMPLETE_BONUS = 1000.0
    PICKUP_BONUS = 50.0
    PROXIMITY_BONUS_SCALE = 0.01
    PROXIMITY_RADIUS = 200.0
    PATH_DIST_DELTA_SCALE = 0.1
    VELOCITY_TOWARD_SCALE = 0.05
    TTI_THRESHOLD = 0.4
    TTI_PENALTY_SCALE = 5.0
    OVERSPEED_PENALTY_SCALE = 0.002
    THRUST_ACCEL = 400.0  # from real_physics.rs

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

        pickup_states = list(env.get_pickup_states())
        path_dist, dir_x, dir_y = self._pathfinder.nearest_pickup_info(
            float(obs[0]), float(obs[1]), pickup_states
        )
        if self._prev_path_dist is not None and self._prev_path_dist > 0:
            reward += (self._prev_path_dist - path_dist) * self.PATH_DIST_DELTA_SCALE
        self._prev_path_dist = path_dist

        if path_dist < self.PROXIMITY_RADIUS:
            reward += (self.PROXIMITY_RADIUS - path_dist) * self.PROXIMITY_BONUS_SCALE

        ship_vx, ship_vy = float(obs[2]), float(obs[3])
        speed = math.sqrt(ship_vx ** 2 + ship_vy ** 2)
        if speed > 1e-6 and (abs(dir_x) > 1e-6 or abs(dir_y) > 1e-6):
            speed_toward = ship_vx * dir_x + ship_vy * dir_y
            if speed_toward > 0:
                reward += speed_toward * self.VELOCITY_TOWARD_SCALE

        if path_dist > 0 and speed > 0:
            v_safe = math.sqrt(2.0 * self.THRUST_ACCEL * path_dist)
            if speed > v_safe:
                reward -= (speed - v_safe) * self.OVERSPEED_PENALTY_SCALE

        min_tti = compute_min_tti(obs)
        if min_tti < self.TTI_THRESHOLD:
            reward -= (self.TTI_THRESHOLD - min_tti) * self.TTI_PENALTY_SCALE

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
