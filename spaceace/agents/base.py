"""Base agent interface + plugin registry.

Registry lives here (not in `spaceace.agents.__init__`) so agents can import
`register_agent` during module-load without triggering a circular import.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import numpy as np

from spaceace.core.env import SpaceAceDirectEnv


AGENT_REGISTRY: dict[str, type["BaseAgent"]] = {}


def register_agent(name: str):
    def deco(cls: type["BaseAgent"]) -> type["BaseAgent"]:
        AGENT_REGISTRY[name] = cls
        return cls

    return deco


class BaseAgent(ABC):
    """
    Minimal agent interface. An agent selects actions for a SpaceAce environment.

    Subclasses own their environment setup — PPO wraps it in VecNormalize,
    MCTS uses the raw direct env, etc. The runner only calls step()
    and get_raw_env() for visualization.
    """

    @abstractmethod
    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        """Initialize the agent's environment(s) and load any models."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset for a new episode."""
        ...

    @abstractmethod
    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Select action and advance one step.

        Returns (action_taken, reward, terminated, truncated, info).
        """
        ...

    @abstractmethod
    def get_raw_env(self) -> SpaceAceDirectEnv:
        """Return the underlying SpaceAceDirectEnv for visualization queries."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        pass
