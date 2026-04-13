"""Human-controlled agent for SpaceAce — play with keyboard."""

from typing import Tuple, Dict, Any

import numpy as np
import pygame

from spaceace.agents.base import BaseAgent, register_agent
from spaceace.core.env import SpaceAceDirectEnv


@register_agent("human")
class HumanAgent(BaseAgent):
    """Arrow keys to rotate, Ctrl to thrust."""

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        self._env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    def reset(self) -> None:
        self._env.reset()

    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        keys = pygame.key.get_pressed()
        rotate_left = keys[pygame.K_LEFT]
        rotate_right = keys[pygame.K_RIGHT]
        thrust = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        action = np.array([int(rotate_left), int(rotate_right), int(thrust)], dtype=np.int32)
        obs, reward, terminated, truncated, info = self._env.step(action)
        return action, reward, terminated, truncated, info

    def get_raw_env(self) -> SpaceAceDirectEnv:
        return self._env

    def close(self) -> None:
        self._env.close()
