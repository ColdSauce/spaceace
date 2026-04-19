"""AlphaZero curriculum training — PPO-style pickup-variant progression.

For each base level, trains on base -> +1 -> +2 -> +3 pickup variants before
advancing. Each stage must hit `advance_win_rate` after at least `min_iters`
self-play iterations to advance. Shares curriculum construction with PPO
(`spaceace.training.curriculum.build_curriculum`).
"""

from __future__ import annotations

import os

# Cap BLAS thread pools before numpy/torch import — M1 Accelerate otherwise
# spawns threads per subproc and contends with our self-play workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import multiprocessing
from pathlib import Path

from spaceace.training.curriculum import build_curriculum, ensure_pickup_variants
from spaceace.training.trainer import AlphaZeroHparams, TrainingConfig

DEFAULT_BASE_LEVELS = list(range(3000, 3165))


def parse_args():
    p = argparse.ArgumentParser(description="AlphaZero curriculum training for SpaceAce")
    p.add_argument("--base-levels", type=str, default=None,
                   help="Comma-separated base level numbers (default: 3000-3164)")
    p.add_argument("--iterations", type=int, default=500,
                   help="Total self-play iterations across the whole curriculum")
    p.add_argument("--iters-per-stage", type=int, default=10,
                   help="Hard cap on iterations per stage before forcing advance")
    p.add_argument("--advance-win-rate", type=float, default=0.7,
                   help="Smoothed win rate required to advance a stage")
    p.add_argument("--min-iters", type=int, default=2,
                   help="Minimum iterations on a stage before it is eligible to advance")
    p.add_argument("--games-per-iter", type=int, default=100)
    p.add_argument("--num-sims", type=int, default=400)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--max-episode-steps", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--buffer-size", type=int, default=5,
                   help="Keep self-play examples from last N iterations")
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from", type=str, default=None,
                   help="Resume from iteration N (loads best_model.pt)")
    p.add_argument("--fresh", action="store_true",
                   help="Delete previous training data and models before starting")
    p.add_argument("--skip-variant-generation", action="store_true",
                   help="Don't auto-generate missing pickup variants")
    return p.parse_args()


def main():
    from spaceace.agents.alphazero.trainer import AlphaZeroTrainer

    args = parse_args()

    if args.base_levels:
        base_levels = [int(x) for x in args.base_levels.split(",")]
    else:
        base_levels = DEFAULT_BASE_LEVELS

    if not args.skip_variant_generation:
        ensure_pickup_variants(base_levels)

    stages = build_curriculum(
        base_levels,
        advance_win_rate=args.advance_win_rate,
        min_iters=args.min_iters,
        max_episode_steps=args.max_episode_steps,
    )

    print(f"Curriculum: {len(base_levels)} levels x 4 stages = {len(stages)} total stages")
    print(f"Progression per level: base -> +1 -> +2 -> +3 pickups")
    print(f"Advance when smoothed win rate >= {args.advance_win_rate:.0%} "
          f"(after >= {args.min_iters} iters, hard cap {args.iters_per_stage})")
    print()

    config = TrainingConfig(
        max_episode_steps=args.max_episode_steps,
        action_repeat=args.action_repeat,
        seed=args.seed,
        model_dir=Path("models/alphazero"),
        curriculum=stages,
        resume_from=args.resume_from,
        alphazero=AlphaZeroHparams(
            iterations=args.iterations,
            games_per_iteration=args.games_per_iter,
            simulations_per_move=args.num_sims,
            c_puct=args.c_puct,
            replay_buffer_shards=args.buffer_size,
            network_train_epochs=args.epochs,
            network_batch_size=args.batch_size,
            network_lr=args.lr,
            eval_games=args.eval_games,
            iters_per_level=args.iters_per_stage,
            win_threshold=args.advance_win_rate,
            fresh=args.fresh,
        ),
    )

    AlphaZeroTrainer().fit(config)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
