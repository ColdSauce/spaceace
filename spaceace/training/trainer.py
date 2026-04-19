"""Trainer ABC + TrainingConfig. Agent-specific trainers subclass this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LevelStage:
    """One stage in a curriculum: a set of levels the agent must master."""

    levels: list[int]
    max_episode_steps: int | None = None
    advance_win_rate: float = 0.7
    min_steps: int = 50_000
    min_iters: int = 1


@dataclass
class AlphaZeroHparams:
    """AlphaZero-specific hyperparameters. Ignored by PPO/SB3 trainers."""

    iterations: int = 100
    games_per_iteration: int = 200
    simulations_per_move: int = 200
    c_puct: float = 1.5
    replay_buffer_shards: int = 50
    network_train_epochs: int = 10
    network_batch_size: int = 256
    network_lr: float = 1e-3
    eval_games: int = 20
    win_rate_window: int = 3
    iters_per_level: int = 10
    win_threshold: float = 0.5
    generate_curriculum: bool = False
    fresh: bool = False


@dataclass
class TrainingConfig:
    """Settings a Trainer consumes. Strategy keys resolve against STRATEGY_REGISTRY."""

    level: int = 0
    total_steps: int = 500_000
    n_envs: int = 16  # paired with DummyVecEnv + [64,64] net; tuned on M1
    learning_rate: float = 3e-4
    max_episode_steps: int = 3000
    action_repeat: int = 5
    seed: int = 42

    obs: str = "path_augmented"
    reward: str = "dense_shaped"
    actions: str = "discrete6"
    pathfinder_backend: str = "grid"

    model_dir: Path = field(default_factory=lambda: Path("./models"))
    tensorboard_dir: Path = field(default_factory=lambda: Path("./tensorboard_logs"))
    eval_freq: int = 25_000
    eval_episodes: int = 10
    resume_from: str | None = None

    curriculum: list[LevelStage] | None = None
    calibration_cache_path: Path | None = None

    alphazero: AlphaZeroHparams = field(default_factory=AlphaZeroHparams)

    @property
    def save_dir(self) -> Path:
        if self.curriculum is not None:
            return self.model_dir / "curriculum"
        return self.model_dir / str(self.level)


class Trainer(ABC):
    """A training run. Subclasses wire a specific RL algorithm to the strategy layer."""

    @abstractmethod
    def fit(self, config: TrainingConfig) -> Path:
        """Train the agent and return the path of the final saved model."""
