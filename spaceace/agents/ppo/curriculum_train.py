"""PPO curriculum training — CLI shim over Sb3Trainer.

All training logic now lives in spaceace.training.sb3_trainer; this module
just parses CLI args into a TrainingConfig and calls fit().

Curriculum structure: for each level, the agent trains on the base level
first (original pickups), then +1, +2, +3 extra pickups before advancing
to the next harder level.

Level numbering: base levels are 3000-3164 (165 total: 25 per strategy
+ 10 transitional levels between each pair). Pickup variants live at
    5000 + (base - 3000) * 3 + (extra - 1)
So L3000 base, L5000 (+1), L5001 (+2), L5002 (+3),
then L3001 base, L5003 (+1), ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spaceace.training.sb3_trainer import Sb3Trainer
from spaceace.training.trainer import LevelStage, TrainingConfig

SOURCE_START = 3000
VARIANT_START = 5000
PICKUPS_PER_LEVEL = 3  # +1 through +3


def pickup_variant(base_level: int, extra: int) -> int:
    """Return the level number for base_level with `extra` additional pickups."""
    idx = base_level - SOURCE_START
    return VARIANT_START + idx * PICKUPS_PER_LEVEL + (extra - 1)


def build_curriculum(
    base_levels: list[int],
    advance_win_rate: float = 0.7,
    min_steps: int = 50_000,
    max_episode_steps: int | None = None,
) -> list[LevelStage]:
    """Build a 2D curriculum: for each level, progress through pickup counts.

    Each stage is one (level, pickup_count) pair. The agent must hit
    advance_win_rate on that stage before moving on.
    """
    stages: list[LevelStage] = []
    for base in base_levels:
        # Stage 0: the original level (no extra pickups)
        stages.append(LevelStage(
            levels=[base],
            max_episode_steps=max_episode_steps,
            advance_win_rate=advance_win_rate,
            min_steps=min_steps,
        ))
        # Stages 1-3: same level with +1 through +3 pickups
        for extra in range(1, PICKUPS_PER_LEVEL + 1):
            stages.append(LevelStage(
                levels=[pickup_variant(base, extra)],
                max_episode_steps=max_episode_steps,
                advance_win_rate=advance_win_rate,
                min_steps=min_steps,
            ))
    return stages


# Default: all 165 base levels, each with 4 stages (base + 3 pickup variants)
DEFAULT_BASE_LEVELS = list(range(3000, 3165))


def parse_args():
    p = argparse.ArgumentParser(description="PPO curriculum training for SpaceAce")
    p.add_argument("--base-levels", type=str, default=None,
                   help="Comma-separated base level numbers (default: 3000-3049)")
    p.add_argument("--timesteps", type=int, default=10_000_000)
    p.add_argument("--advance-win-rate", type=float, default=0.7)
    p.add_argument("--min-steps-per-stage", type=int, default=50_000)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.base_levels:
        base_levels = [int(x) for x in args.base_levels.split(",")]
    else:
        base_levels = DEFAULT_BASE_LEVELS

    stages = build_curriculum(
        base_levels,
        advance_win_rate=args.advance_win_rate,
        min_steps=args.min_steps_per_stage,
    )

    print(f"Curriculum: {len(base_levels)} levels x 4 stages = {len(stages)} total stages")
    print(f"Progression per level: base -> +1 -> +2 -> +3 pickups")
    print()

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
