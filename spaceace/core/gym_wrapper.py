"""Gymnasium wrapper for SpaceAce — standard RL interface for stable-baselines3 etc."""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple, Optional

from spaceace.core.env import SpaceAceDirectEnv


class SpaceAceGymWrapper(gym.Env):
    """Gymnasium-compatible wrapper around SpaceAceDirectEnv."""

    def __init__(self, level: int = 1, max_steps: int = 3000):
        super().__init__()
        self.level = level
        self.max_steps = max_steps
        self.env = SpaceAceDirectEnv(level=level, max_steps=max_steps)

        self.action_space = gym.spaces.MultiDiscrete([2, 2, 2])
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        self.metadata = {'render_modes': ['ascii', 'detailed'], 'render_fps': 60}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        return self.env.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if isinstance(action, (list, tuple)):
            action = np.array(action, dtype=np.int32)
        elif not isinstance(action, np.ndarray):
            action = np.array([action], dtype=np.int32)
        action = np.clip(action, 0, 1).astype(np.int32)
        if len(action) != 3:
            raise ValueError(f"Action must be length 3, got {len(action)}")
        return self.env.step(action)

    def render(self, mode: str = 'ascii') -> str:
        if mode == 'ascii':
            return self.env.render()['ascii_render']
        elif mode == 'detailed':
            return self.env.render()['detailed_render']
        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        self.env.close()

    def get_level_info(self) -> str:
        return self.env.get_level_info()

    def get_map_geometry(self) -> Dict[str, Any]:
        return self.env.get_map_geometry()

    def get_pickup_states(self):
        return self.env.get_pickup_states()
