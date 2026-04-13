"""PPO training entry point — delegates to the generic Sb3Trainer.

Kept so `python -m spaceace.agents.ppo.train` still works. The real work
lives in `spaceace.training.sb3_trainer`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spaceace.training import Sb3Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a SpaceAce PPO agent")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Default: 1500 for level 0, 3000 otherwise")
    p.add_argument("--n-envs", type=int, default=0, help="0 = cpu_count()")
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--tensorboard-dir", type=str, default="./tensorboard_logs/")
    p.add_argument("--model-dir", type=str, default="./models/")
    p.add_argument("--obs", type=str, default="path_augmented")
    p.add_argument("--reward", type=str, default="dense_shaped")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_steps = args.max_steps if args.max_steps is not None else (1500 if args.level == 0 else 3000)
    config = TrainingConfig(
        level=args.level,
        total_steps=args.timesteps,
        n_envs=args.n_envs,
        max_episode_steps=max_steps,
        action_repeat=args.action_repeat,
        seed=args.seed,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        model_dir=Path(args.model_dir),
        tensorboard_dir=Path(args.tensorboard_dir),
        obs=args.obs,
        reward=args.reward,
        resume_from=args.resume,
    )
    Sb3Trainer().fit(config)


if __name__ == "__main__":
    main()
