"""Train the low-level waypoint pilot for the HRL agent using PPO.

CLI entry point. The actual training logic lives in HrlTrainer.

Usage:
    uv run python -m spaceace.agents.hrl.train_pilot --levels 4000-4099 --timesteps 500000
    uv run python -m spaceace.agents.hrl.train_pilot --levels 4000 4010 4020 --timesteps 100000
"""

import argparse
from pathlib import Path


def parse_level_spec(spec: str) -> list[int]:
    """Parse level spec like '4000-4099' or '4000' into list of ints."""
    if "-" in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(spec)]


def parse_args():
    p = argparse.ArgumentParser(description="Train waypoint pilot for HRL agent (PPO)")
    p.add_argument("--levels", type=str, nargs="+", default=["4000-4099"],
                   help="Level specs (e.g., 4000-4099 or 4000 4010)")
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Total training timesteps (default: 2000000)")
    p.add_argument("--max-steps", type=int, default=500,
                   help="Max steps per waypoint episode (default: 500)")
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="Evaluate every N timesteps (default: 10000)")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="Episodes per evaluation (default: 20)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--tensorboard-dir", type=str, default="./tensorboard_logs/",
                   help="TensorBoard log dir")
    p.add_argument("--model-dir", type=str, default="./models/hrl/pilot/",
                   help="Model save dir")
    p.add_argument("--action-repeat", type=int, default=2,
                   help="Frames per action (default: 2)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to model to resume training from (without .zip)")
    return p.parse_args()


def main():
    from spaceace.agents.hrl.trainer import HrlTrainer
    from spaceace.training.trainer import TrainingConfig

    args = parse_args()

    levels = []
    for spec in args.levels:
        levels.extend(parse_level_spec(spec))
    levels = sorted(set(levels))

    config = TrainingConfig(
        level=levels[0],
        total_steps=args.timesteps,
        n_envs=min(16, max(8, len(levels))),
        max_episode_steps=args.max_steps,
        action_repeat=args.action_repeat,
        seed=args.seed,
        model_dir=Path(args.model_dir),
        tensorboard_dir=Path(args.tensorboard_dir),
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        resume_from=args.resume,
    )

    trainer = HrlTrainer(levels=levels)
    trainer.fit(config)


if __name__ == "__main__":
    main()
