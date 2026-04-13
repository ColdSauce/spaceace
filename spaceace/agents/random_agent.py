"""Random baseline agent for SpaceAce."""

from typing import Tuple, Dict, Any

import numpy as np

from spaceace.agents.base import BaseAgent, register_agent
from spaceace.core.env import SpaceAceDirectEnv


@register_agent("random")
class RandomAgent(BaseAgent):
    """Selects random actions each step. Useful as a baseline."""

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        self._env = SpaceAceDirectEnv(level=level, max_steps=max_steps)

    def reset(self) -> None:
        self._env.reset()

    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.random.randint(0, 2, size=3).astype(np.int32)
        obs, reward, terminated, truncated, info = self._env.step(action)
        return action, reward, terminated, truncated, info

    def get_raw_env(self) -> SpaceAceDirectEnv:
        return self._env

    def close(self) -> None:
        self._env.close()
