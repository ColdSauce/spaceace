"""PPO curriculum training — CLI shim over Sb3Trainer.

All training logic now lives in spaceace.training.sb3_trainer; this module
just parses CLI args into a TrainingConfig and calls fit().
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spaceace.training.sb3_trainer import Sb3Trainer
from spaceace.training.trainer import LevelStage, TrainingConfig

STAGE_SIZE = 5
DEFAULT_LEVELS = list(range(3000, 3050)) + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 100]


def parse_args():
    p = argparse.ArgumentParser(description="PPO curriculum training for SpaceAce")
    p.add_argument("--stages", type=str, default=None,
                   help="Comma-separated level numbers (grouped into stages of 5)")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--advance-win-rate", type=float, default=0.5)
    p.add_argument("--min-steps-per-stage", type=int, default=200_000)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.stages:
        levels = [int(x) for x in args.stages.split(",")]
    else:
        levels = DEFAULT_LEVELS

    # Group into stages of STAGE_SIZE
    stages = []
    for i in range(0, len(levels), STAGE_SIZE):
        stages.append(LevelStage(
            levels=levels[i : i + STAGE_SIZE],
            max_episode_steps=None,  # calibrate via MCTS
            advance_win_rate=args.advance_win_rate,
            min_steps=args.min_steps_per_stage,
        ))

    config = TrainingConfig(
        total_steps=args.timesteps,
        action_repeat=args.action_repeat,
        seed=args.seed,
        curriculum=stages,
        calibration_cache_path=Path("data/calibration_cache.json"),
        resume_from=args.resume_from,
    )

    Sb3Trainer().fit(config)


if __name__ == "__main__":
    main()
