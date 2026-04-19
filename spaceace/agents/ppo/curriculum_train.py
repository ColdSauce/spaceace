"""PPO curriculum training — CLI shim over Sb3Trainer.

All training logic now lives in spaceace.training.sb3_trainer; this module
just parses CLI args into a TrainingConfig and calls fit().

Curriculum structure: for each level, the agent trains on the base level
first (original pickups), then +1, +2, +3 extra pickups before advancing
to the next harder level.

Level numbering: base levels are 3000-3164 (165 total: 25 per strategy
+ 10 transitional levels between each pair). Pickup variants live at
    5000 + (base - 3000) * VARIANT_STRIDE + (extra - 1)
where VARIANT_STRIDE must match `add_pickups.py --max-extra` (default 5).
So L3000 base, L5000 (+1) ... L5004 (+5),
then L3001 base, L5005 (+1) ..., etc.
"""

from __future__ import annotations

import os

# Cap BLAS thread pools before numpy/torch import — M1 Accelerate will otherwise
# spawn threads per subproc and contend with our SubprocVecEnv workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from pathlib import Path

from spaceace.training.curriculum import (
    PICKUPS_PER_LEVEL,
    SOURCE_START,
    VARIANT_START,
    VARIANT_STRIDE,
    build_curriculum,
    pickup_variant,
)
from spaceace.training.sb3_trainer import Sb3Trainer
from spaceace.training.trainer import LevelStage, TrainingConfig

# Default: all 165 base levels, each with 4 stages (base + 3 pickup variants)
DEFAULT_BASE_LEVELS = list(range(3000, 3165))


def parse_args():
    p = argparse.ArgumentParser(description="PPO curriculum training for SpaceAce")
    p.add_argument("--base-levels", type=str, default=None,
                   help="Comma-separated base level numbers (default: 3000-3049)")
    p.add_argument("--timesteps", type=int, default=50_000_000)
    p.add_argument("--advance-win-rate", type=float, default=0.7)
    p.add_argument("--min-steps-per-stage", type=int, default=50_000)
    p.add_argument("--max-episode-steps", type=int, default=3000,
                   help="Per-stage step cap. Pass 0 to enable MCTS calibration.")
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--n-envs", type=int, default=16,
                   help="Parallel envs. Default 16 pairs with DummyVecEnv + [64,64] net.")
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
        max_episode_steps=args.max_episode_steps if args.max_episode_steps > 0 else None,
    )

    print(f"Curriculum: {len(base_levels)} levels x 4 stages = {len(stages)} total stages")
    print(f"Progression per level: base -> +1 -> +2 -> +3 pickups")
    print()

    config = TrainingConfig(
        total_steps=args.timesteps,
        n_envs=args.n_envs,
        action_repeat=args.action_repeat,
        seed=args.seed,
        curriculum=stages,
        calibration_cache_path=Path("data/calibration_cache.json"),
        resume_from=args.resume_from,
    )

    Sb3Trainer().fit(config)


if __name__ == "__main__":
    main()
