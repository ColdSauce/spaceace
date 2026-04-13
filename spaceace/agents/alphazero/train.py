"""AlphaZero training loop: self-play -> train -> evaluate -> repeat.

CLI entry point. The actual training logic lives in AlphaZeroTrainer.
Run directly or via ``python -m spaceace.agents.alphazero.train``.
"""

import argparse
import multiprocessing
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from spaceace.agents.alphazero.network import AlphaZeroNet, export_to_onnx
from spaceace.agents.alphazero.self_play import (
    run_self_play, save_examples, load_examples, GameExample, GameStats,
)


def parse_args():
    p = argparse.ArgumentParser(description="AlphaZero training for SpaceAce")
    p.add_argument("--level", type=int, default=None,
                   help="Single level to train on (overrides --curriculum)")
    p.add_argument("--curriculum", type=int, nargs="+", default=None,
                   help="Levels in order of difficulty")
    p.add_argument("--generate-curriculum", action="store_true",
                   help="Auto-generate curriculum maps (levels 3000-3049) before training")
    p.add_argument("--iters-per-level", type=int, default=10,
                   help="Max iterations per level before advancing (default: 10)")
    p.add_argument("--win-threshold", type=float, default=0.5,
                   help="Win rate to advance to next level (default: 0.5)")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--games-per-iter", type=int, default=100)
    p.add_argument("--num-sims", type=int, default=400)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--buffer-size", type=int, default=5,
                   help="Keep examples from last N iterations")
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--resume", type=int, default=0,
                   help="Resume from iteration N")
    p.add_argument("--fresh", action="store_true",
                   help="Delete previous training data and models before starting")
    return p.parse_args()


# Generated curriculum levels (3000-3049) with per-stage settings
CURRICULUM_STAGES = [
    # (level_range, max_steps, num_sims_multiplier, description)
    (range(3000, 3010), 500, 5, "simple"),        # 5x sims to bootstrap wins
    (range(3010, 3020), 1500, 2, "room"),          # 2x sims
    (range(3020, 3030), 2000, 1, "maze"),          # normal sims
    (range(3030, 3040), 3000, 1, "cave"),
    (range(3040, 3050), 3000, 1, "gauntlet"),
]

DEFAULT_CURRICULUM = list(range(3000, 3050))


def get_level_settings(level: int, base_max_steps: int = 3000, base_sims: int = 400) -> tuple[int, int]:
    """Return (max_steps, num_sims) for a level based on its difficulty stage."""
    for level_range, max_steps, sims_mult, _ in CURRICULUM_STAGES:
        if level in level_range:
            return max_steps, base_sims * sims_mult
    return base_max_steps, base_sims


def generate_curriculum_maps():
    """Generate curriculum maps using generate_maps.py."""
    configs = [
        ("simple", 3000, 100),
        ("room", 3010, 200),
        ("maze", 3020, 300),
        ("cave", 3030, 400),
        ("gauntlet", 3040, 500),
    ]
    for strategy, start_level, seed in configs:
        print(f"Generating {strategy} maps (levels {start_level}-{start_level+9})...")
        subprocess.run([
            "uv", "run", "python", "generate_maps.py",
            "--count", "10", "--strategy", strategy,
            "--start-level", str(start_level), "--seed", str(seed), "--merge",
        ], check=True)
    print()


def train_network(
    model: AlphaZeroNet,
    examples: list[GameExample],
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    batch_size: int = 256,
) -> dict:
    """Train the network on collected examples. Returns loss metrics."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.train()

    observations = torch.tensor(np.array([e.observation for e in examples]), dtype=torch.float32)
    target_policies = torch.tensor(np.array([e.mcts_policy for e in examples]), dtype=torch.float32)
    target_values = torch.tensor(np.array([e.value_target for e in examples]), dtype=torch.float32)

    dataset = TensorDataset(observations, target_policies, target_values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_batches = 0

    for epoch in range(epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_batches = 0

        for obs_batch, pol_batch, val_batch in loader:
            obs_batch = obs_batch.to(device)
            pol_batch = pol_batch.to(device)
            val_batch = val_batch.to(device)

            pred_policy, pred_value = model(obs_batch)

            # Policy loss: cross-entropy with MCTS policy targets
            policy_loss = -torch.sum(pol_batch * F.log_softmax(pred_policy, dim=1)) / obs_batch.shape[0]
            # Value loss: MSE
            value_loss = F.mse_loss(pred_value, val_batch)

            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_batches += 1

        total_policy_loss += epoch_policy_loss
        total_value_loss += epoch_value_loss
        total_batches += epoch_batches

    model.cpu()

    return {
        "policy_loss": total_policy_loss / max(total_batches, 1),
        "value_loss": total_value_loss / max(total_batches, 1),
    }


def evaluate_model(
    level: int,
    model_path: str,
    num_games: int = 20,
    num_sims: int = 200,
    action_repeat: int = 5,
    max_steps: int = 3000,
) -> dict:
    """Evaluate a model by playing games. Returns reward, pickup, and win stats."""
    _, stats = run_self_play(
        level, num_games, num_sims,
        action_repeat=action_repeat, max_steps=max_steps,
        model_path=model_path,
    )

    n = len(stats)
    wins = sum(1 for s in stats if s.completed)
    crashes = sum(1 for s in stats if s.crashed)
    mean_reward = sum(s.total_reward for s in stats) / max(n, 1)
    mean_pickups = sum(s.pickups_collected for s in stats) / max(n, 1)

    return {
        "wins": wins,
        "crashes": crashes,
        "total": n,
        "win_rate": wins / max(n, 1),
        "mean_reward": mean_reward,
        "mean_pickups": mean_pickups,
    }


def main():
    from pathlib import Path
    from spaceace.agents.alphazero.trainer import AlphaZeroTrainer
    from spaceace.training.trainer import AlphaZeroHparams, LevelStage, TrainingConfig

    args = parse_args()

    # Build curriculum from CLI args
    curriculum = None
    level = args.level if args.level is not None else 0
    if args.curriculum is not None:
        curriculum = [LevelStage(levels=args.curriculum)]
    elif args.level is None:
        curriculum = [LevelStage(levels=DEFAULT_CURRICULUM)]

    config = TrainingConfig(
        level=level,
        max_episode_steps=args.max_steps,
        action_repeat=args.action_repeat,
        model_dir=Path("models/alphazero"),
        resume_from=str(args.resume) if args.resume > 0 else None,
        curriculum=curriculum,
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
            iters_per_level=args.iters_per_level,
            win_threshold=args.win_threshold,
            generate_curriculum=args.generate_curriculum,
            fresh=args.fresh,
        ),
    )

    trainer = AlphaZeroTrainer()
    trainer.fit(config)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
