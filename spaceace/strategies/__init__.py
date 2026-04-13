"""Composable building blocks agents consume.

Strategies are stateful objects owned by one env; they encapsulate pathfinding,
observation shaping, reward shaping, and action encoding so agents and trainers
do not reinvent them. See `spaceace.strategies.base` for interfaces.
"""

from spaceace.strategies.base import (
    ActionSpace,
    ObservationBuilder,
    Pathfinder,
    RewardShaper,
    StrategyBundle,
)
from spaceace.strategies.actions import DiscreteAction6
from spaceace.strategies.observation import PathAugmentedObs23, RawObs19
from spaceace.strategies.pathfinder import RustPathfinder
from spaceace.strategies.rewards import DenseShapedReward, SparseReward

STRATEGY_REGISTRY: dict[str, dict[str, type]] = {
    "pathfinder": {"rust": RustPathfinder},
    "observation": {"raw": RawObs19, "path_augmented": PathAugmentedObs23},
    "reward": {"sparse": SparseReward, "dense_shaped": DenseShapedReward},
    "actions": {"discrete6": DiscreteAction6},
}


def resolve(kind: str, key: str):
    return STRATEGY_REGISTRY[kind][key]


__all__ = [
    "ActionSpace",
    "DiscreteAction6",
    "ObservationBuilder",
    "PathAugmentedObs23",
    "Pathfinder",
    "RawObs19",
    "RewardShaper",
    "RustPathfinder",
    "DenseShapedReward",
    "SparseReward",
    "STRATEGY_REGISTRY",
    "StrategyBundle",
    "resolve",
]
