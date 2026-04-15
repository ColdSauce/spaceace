"""SpaceAce Direct Environment — Calls Rust game engine via PyO3 (no IPC overhead)."""

import numpy as np
from typing import Tuple, Dict, Any

import spaceace_rl  # compiled Rust PyO3 module


class SpaceAceDirectEnv:
    """Direct Python-to-Rust game environment. No subprocess, no JSON."""

    def __init__(self, level: int = 1, max_steps: int = 3000):
        self.level = level
        self.max_steps = max_steps
        self.observation_space_size = 20
        self.action_space_size = 3
        self._game = spaceace_rl.PyGameInstance(level, max_steps)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs = self._game.reset()
        info = dict(self._game.get_info())
        return np.array(obs, dtype=np.float32), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if len(action) != 3:
            raise ValueError("Action must be length 3: [rotate_left, rotate_right, thrust]")
        obs, reward, terminated, truncated, info = self._game.step(
            action.astype(int).tolist()
        )
        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, dict(info)

    def get_observation(self) -> np.ndarray:
        return np.array(self._game.get_observation(), dtype=np.float32)

    def get_info(self) -> Dict[str, Any]:
        return dict(self._game.get_info())

    def get_level_info(self) -> str:
        return self._game.get_level_info()

    def render(self) -> Dict[str, str]:
        return {
            "ascii_render": self._game.render_ascii(),
            "detailed_render": self._game.render_detailed(),
        }

    def get_pickup_states(self):
        """Return list of booleans: True if pickup is collected."""
        return self._game.get_pickup_states()

    def save_state(self):
        """Return an opaque state snapshot for tree search."""
        return self._game.save_state()

    def load_state(self, state):
        """Restore a previously saved state snapshot."""
        self._game.load_state(state)

    def get_map_geometry(self) -> Dict[str, Any]:
        return dict(self._game.get_map_geometry())

    def close(self):
        pass
