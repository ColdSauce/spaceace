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
    p.add_argument("--buffer-size", type=int, default=30,
                   help="Keep examples from last N iterations")
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--resume", type=int, default=0,
                   help="Resume from iteration N")
    p.add_argument("--fresh", action="store_true",
                   help="Delete previous training data and models before starting")
    return p.parse_args()


DEFAULT_BASE_LEVELS = list(range(3000, 3165))


def generate_curriculum_maps():
    """Generate curriculum maps using generate_maps.py."""
    configs = [
        # (strategy, start_level, count, seed)
        ("simple",           3000, 25, 100),
        ("simple_to_room",   3025, 10, 150),
        ("room",             3035, 25, 200),
        ("room_to_maze",     3060, 10, 250),
        ("maze",             3070, 25, 300),
        ("maze_to_cave",     3095, 10, 350),
        ("cave",             3105, 25, 400),
        ("cave_to_gauntlet", 3130, 10, 450),
        ("gauntlet",         3140, 25, 500),
    ]
    for strategy, start_level, count, seed in configs:
        end_level = start_level + count - 1
        print(f"Generating {strategy} maps (levels {start_level}-{end_level})...")
        subprocess.run([
            "uv", "run", "python", "generate_maps.py",
            "--count", str(count), "--strategy", strategy,
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.train()

    observations = torch.tensor(np.array([e.observation for e in examples]), dtype=torch.float32)
    target_policies = torch.tensor(np.array([e.mcts_policy for e in examples]), dtype=torch.float32)
    target_values = torch.tensor(np.array([e.value_target for e in examples]), dtype=torch.float32)

    # Diagnostics: value/policy target distributions tell you whether the
    # signal is alive. If val_std collapses to ~0 or pol_entropy is flat at
    # log(6)=1.79, the network has no useful gradient to follow.
    val_mean = float(target_values.mean())
    val_std = float(target_values.std())
    val_min = float(target_values.min())
    val_max = float(target_values.max())
    frac_nonzero = float((target_values.abs() > 1e-6).float().mean())
    eps = 1e-9
    pol_entropy = float(
        -(target_policies * (target_policies + eps).log()).sum(dim=1).mean()
    )
    print(
        f"  Targets: value mean={val_mean:+.3f} std={val_std:.3f} "
        f"range=[{val_min:+.3f},{val_max:+.3f}] nonzero={frac_nonzero:.2f} | "
        f"policy entropy={pol_entropy:.3f} (uniform={np.log(target_policies.shape[1]):.3f})"
    )

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

            log_pred = F.log_softmax(pred_policy, dim=1)
            # Policy loss: KL to MCTS visit distribution.
            policy_loss = F.kl_div(log_pred, pol_batch, reduction="batchmean")
            # Entropy bonus: keep policy from collapsing before MCTS can explore.
            pred_probs = log_pred.exp()
            entropy = -(pred_probs * log_pred).sum(dim=1).mean()
            # Value loss: MSE, downweighted (AZ paper uses 0.5 to stop value head
            # from dominating when targets cluster tightly at the heuristic).
            value_loss = F.mse_loss(pred_value, val_batch)

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

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
    from spaceace.training.curriculum import build_curriculum, ensure_pickup_variants
    from spaceace.training.trainer import AlphaZeroHparams, LevelStage, TrainingConfig

    args = parse_args()

    # Build curriculum from CLI args.
    # - `--level N`: single-stage curriculum on L{N}
    # - `--curriculum a b c`: one stage per listed level
    # - otherwise: default pickup-variant curriculum over DEFAULT_BASE_LEVELS
    level = args.level if args.level is not None else 0
    if args.level is not None:
        curriculum = [LevelStage(levels=[args.level], max_episode_steps=args.max_steps)]
    elif args.curriculum is not None:
        curriculum = [
            LevelStage(levels=[lvl], max_episode_steps=args.max_steps)
            for lvl in args.curriculum
        ]
    else:
        ensure_pickup_variants(DEFAULT_BASE_LEVELS)
        curriculum = build_curriculum(
            DEFAULT_BASE_LEVELS,
            advance_win_rate=args.win_threshold,
            max_episode_steps=args.max_steps,
        )

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
