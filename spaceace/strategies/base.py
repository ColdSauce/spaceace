"""Strategy ABCs: Pathfinder, ObservationBuilder, RewardShaper, ActionSpace."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np


class Pathfinder(Protocol):
    """Wall-aware navigation queries for observation/reward shaping."""

    def nearest_pickup_info(
        self, ship_x: float, ship_y: float, collected: list[bool]
    ) -> tuple[float, float, float]:
        """Return (path_distance, direction_x, direction_y) to the best uncollected pickup."""
        ...


class ObservationBuilder(ABC):
    """Transforms the raw 19-dim Rust observation into whatever the agent needs.

    Strategies own episode state (e.g. step counter for time_remaining features)
    so the env wrapper stays a dumb pipe.
    """

    space: gym.spaces.Box

    @abstractmethod
    def reset(self, raw_obs: np.ndarray, info: dict, env) -> np.ndarray: ...

    @abstractmethod
    def build(self, raw_obs: np.ndarray, info: dict, env) -> np.ndarray: ...


class RewardShaper(ABC):
    """Replaces the base environment reward with something denser (or keeps it sparse)."""

    @abstractmethod
    def reset(self, raw_obs: np.ndarray, info: dict, env) -> None: ...

    @abstractmethod
    def shape(
        self, raw_obs: np.ndarray, action: np.ndarray, info: dict, env
    ) -> float: ...

    def episode_metrics(self) -> dict[str, Any]:
        """Optional: metrics to attach to the terminal step's info dict."""
        return {}


class ActionSpace(ABC):
    """Maps agent-space actions to the raw [rot_left, rot_right, thrust] Rust triplet."""

    space: gym.spaces.Space

    @abstractmethod
    def decode(self, action) -> np.ndarray: ...


@dataclass
class StrategyBundle:
    """The four strategies an agent or trainer composes together."""

    pathfinder: Pathfinder | None
    observation: ObservationBuilder
    reward: RewardShaper
    actions: ActionSpace
