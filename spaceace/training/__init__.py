"""Unified training infrastructure shared across agents."""

from spaceace.training.trainer import Trainer, TrainingConfig
from spaceace.training.envs import make_vec_env, make_single_env

TRAINER_REGISTRY: dict[str, type[Trainer]] = {}


def register_trainer(name: str):
    def deco(cls: type[Trainer]) -> type[Trainer]:
        TRAINER_REGISTRY[name] = cls
        return cls

    return deco


from spaceace.training.sb3_trainer import Sb3Trainer  # noqa: E402

TRAINER_REGISTRY["sb3"] = Sb3Trainer
TRAINER_REGISTRY["ppo"] = Sb3Trainer

__all__ = [
    "Trainer",
    "TrainingConfig",
    "TRAINER_REGISTRY",
    "register_trainer",
    "make_vec_env",
    "make_single_env",
    "Sb3Trainer",
]
