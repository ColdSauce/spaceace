"""Observation builders: raw passthrough and pathfinder-augmented variants."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from spaceace.strategies.base import ObservationBuilder, Pathfinder


class RawObs(ObservationBuilder):
    """No-op passthrough of the 36-dim Rust observation.

    Layout: 0..19 ship/pickup/wall-8/min_tti (as before), 20..36 the 16 fine-grained
    wall raycasts (interleaved with the 8 coarse rays for 24 total at 15° spacing).
    """

    def __init__(self) -> None:
        self.space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)

    def reset(self, raw_obs: np.ndarray, info: dict, env) -> np.ndarray:
        return raw_obs.astype(np.float32, copy=False)

    def build(self, raw_obs: np.ndarray, info: dict, env) -> np.ndarray:
        return raw_obs.astype(np.float32, copy=False)


# Keep old name as alias for backwards compatibility with strategy registry.
RawObs19 = RawObs


class PathAugmentedObs23(ObservationBuilder):
    """Drops absolute positions, adds 8 pathfinder/TTI/time features + 16 fine rays. Total 40 dims.

    Composition: obs[2:5] (vx, vy, rot) + obs[7:8] (pickup dist) + obs[8:16] (wall-8)
    + obs[16:19] (pickups remaining, normalized x/y) + 8 derived features
    + obs[20:36] (16 fine wall rays, scaled).
    """

    def __init__(self, pathfinder: Pathfinder, max_steps: int) -> None:
        self.space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
        self._pathfinder = pathfinder
        self._max_steps = max_steps
        self._steps = 0
        self._buf = np.empty(40, dtype=np.float32)

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
        min_tti = float(obs[19])  # Computed in Rust build_observation

        ship_vx = float(obs[2])
        ship_vy = float(obs[3])
        ship_rot = float(obs[4])
        sin_rot = math.sin(ship_rot)
        cos_rot = math.cos(ship_rot)

        buf = self._buf
        # Filtered raw features (16 dims)
        buf[0] = obs[2] / 300.0       # vx
        buf[1] = obs[3] / 300.0       # vy
        buf[2] = sin_rot              # rot sin
        buf[3] = cos_rot              # rot cos
        buf[4] = obs[7] / 1000.0      # pickup distance
        buf[5:13] = obs[8:16]         # wall distances (scaled below)
        buf[5:13] /= 1000.0
        buf[13] = obs[16] / 10.0      # pickups remaining
        buf[14] = obs[17]             # norm_x
        buf[15] = obs[18]             # norm_y

        # Derived features (8 dims)
        speed = math.sqrt(ship_vx ** 2 + ship_vy ** 2)
        if speed > 1e-6 and (abs(dir_x) > 1e-6 or abs(dir_y) > 1e-6):
            speed_toward = (ship_vx * dir_x + ship_vy * dir_y) / speed
        else:
            speed_toward = 0.0

        buf[16] = min(path_dist / 5000.0, 1.0)
        buf[17] = dir_x
        buf[18] = dir_y
        buf[19] = speed / 300.0
        buf[20] = speed_toward / 300.0
        buf[21] = sin_rot * dir_x + (-cos_rot) * dir_y  # heading alignment
        buf[22] = min(min_tti, 2.0) / 2.0
        buf[23] = max(0.0, 1.0 - self._steps / self._max_steps)

        # 16 fine wall rays (indices 24..40), scaled like the coarse 8.
        buf[24:40] = obs[20:36]
        buf[24:40] /= 1000.0

        return buf.copy()
