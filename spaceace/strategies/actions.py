"""Action space strategies. Canonical source of truth for the 6-action set."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

# 6 macro-actions over (rot_left, rot_right, thrust). Order must match Rust ACTIONS.
ALL_ACTIONS: list[np.ndarray] = [
    np.array([0, 0, 0], dtype=np.int32),  # coast
    np.array([0, 0, 1], dtype=np.int32),  # thrust
    np.array([1, 0, 0], dtype=np.int32),  # rotate left
    np.array([1, 0, 1], dtype=np.int32),  # rotate left + thrust
    np.array([0, 1, 0], dtype=np.int32),  # rotate right
    np.array([0, 1, 1], dtype=np.int32),  # rotate right + thrust
]

ACTION_NAMES: list[str] = ["COAST", "THRUST", "LEFT", "LEFT+THR", "RIGHT", "RIGHT+THR"]


class DiscreteAction6:
    """MultiDiscrete([2,2,2]) passthrough — SB3-friendly."""

    def __init__(self) -> None:
        self.space = gym.spaces.MultiDiscrete([2, 2, 2])

    def decode(self, action) -> np.ndarray:
        if isinstance(action, (list, tuple)):
            action = np.array(action, dtype=np.int32)
        elif not isinstance(action, np.ndarray):
            action = np.array([action], dtype=np.int32)
        return np.clip(action, 0, 1).astype(np.int32)
