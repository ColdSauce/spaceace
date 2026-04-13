"""Observation builders: raw passthrough and pathfinder-augmented variants."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from spaceace.strategies.base import ObservationBuilder, Pathfinder

# 8 raycast directions relative to ship heading. Must match Rust BASE_DIRS.
BASE_DIRS: list[tuple[float, float]] = [
    (0.0, -1.0),
    (0.707, -0.707),
    (1.0, 0.0),
    (0.707, 0.707),
    (0.0, 1.0),
    (-0.707, 0.707),
    (-1.0, 0.0),
    (-0.707, -0.707),
]


def compute_min_tti(obs: np.ndarray) -> float:
    """Minimum time-to-impact across 8 raycast directions. Infinity if nothing closes."""
    ship_vx, ship_vy = float(obs[2]), float(obs[3])
    ship_rot = float(obs[4])
    wall_distances = obs[8:16]

    cos_r = math.cos(ship_rot)
    sin_r = math.sin(ship_rot)
    min_tti = float("inf")
    for i, (dx, dy) in enumerate(BASE_DIRS):
        world_dx = dx * cos_r - dy * sin_r
        world_dy = dx * sin_r + dy * cos_r
        v_toward = ship_vx * world_dx + ship_vy * world_dy
        if v_toward > 1.0:
            tti = float(wall_distances[i]) / v_toward
            if tti < min_tti:
                min_tti = tti
    return min_tti


class RawObs19(ObservationBuilder):
    """No-op passthrough of the 19-dim Rust observation."""

    def __init__(self) -> None:
        self.space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)

    def reset(self, raw_obs: np.ndarray, info: dict, env) -> np.ndarray:
        return raw_obs.astype(np.float32, copy=False)

    def build(self, raw_obs: np.ndarray, info: dict, env) -> np.ndarray:
        return raw_obs.astype(np.float32, copy=False)


class PathAugmentedObs23(ObservationBuilder):
    """Drops absolute positions, adds 8 pathfinder/TTI/time features. Total 23 dims.

    Composition: obs[2:5] (vx, vy, rot) + obs[7:8] (pickup dist) + obs[8:16] (wall)
    + obs[16:19] (pickups remaining, normalized x/y) + 8 derived features.
    """

    def __init__(self, pathfinder: Pathfinder, max_steps: int) -> None:
        self.space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)
        self._pathfinder = pathfinder
        self._max_steps = max_steps
        self._steps = 0

    def reset(self, raw_obs: np.ndarray, info: dict, env) -> np.ndarray:
        self._steps = 0
        return self._build(raw_obs, env)

    def build(self, raw_obs: np.ndarray, info: dict, env) -> np.ndarray:
        self._steps += 1
        return self._build(raw_obs, env)

    def _build(self, obs: np.ndarray, env) -> np.ndarray:
        collected = list(env.get_pickup_states())
        path_dist, dir_x, dir_y = self._pathfinder.nearest_pickup_info(
            float(obs[0]), float(obs[1]), collected
        )
        min_tti = compute_min_tti(obs)

        ship_vx, ship_vy, ship_rot = obs[2], obs[3], obs[4]
        filtered = np.concatenate(
            [
                obs[2:5],    # vx, vy, rot
                obs[7:8],    # pickup distance (euclidean)
                obs[8:16],   # 8 wall distances
                obs[16:19],  # pickups_remaining, norm_x, norm_y
            ]
        )

        path_dist_norm = min(path_dist / 5000.0, 1.0)
        speed = math.sqrt(float(ship_vx) ** 2 + float(ship_vy) ** 2)
        if speed > 1e-6 and (abs(dir_x) > 1e-6 or abs(dir_y) > 1e-6):
            speed_toward = (float(ship_vx) * dir_x + float(ship_vy) * dir_y) / speed
        else:
            speed_toward = 0.0
        heading_x = math.sin(float(ship_rot))
        heading_y = -math.cos(float(ship_rot))
        heading_alignment = heading_x * dir_x + heading_y * dir_y
        min_tti_norm = min(min_tti, 2.0) / 2.0
        time_remaining = 1.0 - self._steps / self._max_steps

        extra = np.array(
            [
                path_dist_norm,
                dir_x,
                dir_y,
                speed,
                speed_toward,
                heading_alignment,
                min_tti_norm,
                time_remaining,
            ],
            dtype=np.float32,
        )
        return np.concatenate([filtered, extra]).astype(np.float32, copy=False)
