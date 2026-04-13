"""AlphaZero training loop: self-play → train → evaluate → repeat."""

import argparse
import multiprocessing
import os
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
    import subprocess
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
    args = parse_args()

    # Generate curriculum maps if requested
    if args.generate_curriculum:
        generate_curriculum_maps()

    # Determine level schedule
    if args.level is not None:
        levels = [args.level]
    elif args.curriculum is not None:
        levels = args.curriculum
    else:
        levels = DEFAULT_CURRICULUM

    save_dir = "models/alphazero/curriculum"

    if args.fresh:
        import shutil
        for lvl in levels:
            data_dir = f"data/alphazero/{lvl}"
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        print("Cleared previous training data and models.")
        args.resume = 0

    os.makedirs(save_dir, exist_ok=True)

    print(f"=== AlphaZero Training ===")
    print(f"Curriculum: {len(levels)} levels (L{levels[0]} -> L{levels[-1]})")
    print(f"Win threshold to advance: {args.win_threshold:.0%}")
    print(f"Max iters/level: {args.iters_per_level}")
    print(f"Total iterations: {args.iterations}")
    print(f"Games/iter: {args.games_per_iter}")
    print(f"Sims/move: {args.num_sims}")
    print()

    # Initialize or load model
    model = AlphaZeroNet()
    best_model_pt = os.path.join(save_dir, "best_model.pt")
    best_model_onnx = os.path.join(save_dir, "best_model.onnx")

    if args.resume > 0 and os.path.exists(best_model_pt):
        model.load_state_dict(torch.load(best_model_pt, weights_only=True))
        print(f"Resumed from {best_model_pt}")

    # Create optimizer once — persists across iterations
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    iteration = args.resume
    end_iteration = args.resume + args.iterations
    level_idx = 0  # current position in curriculum
    iters_on_current = 0  # iterations spent on current level
    levels_seen = set()  # track which levels have contributed data
    recent_win_rates: list[float] = []  # sliding window for smoothed advancement
    WIN_RATE_WINDOW = 3  # require consistent wins over this many evals

    # Group levels by stage (same settings = same difficulty tier)
    # Levels within a stage are cycled during self-play for diversity
    def get_stage_levels(level_idx: int) -> list[int]:
        """Get all levels in the same difficulty stage as levels[level_idx]."""
        target = levels[level_idx]
        for level_range, _, _, _ in CURRICULUM_STAGES:
            if target in level_range:
                return [l for l in levels if l in level_range]
        return [target]

    # Data dirs
    for lvl in levels:
        os.makedirs(f"data/alphazero/{lvl}", exist_ok=True)

    while iteration < end_iteration and level_idx < len(levels):
        current_level = levels[level_idx]
        stage_levels = get_stage_levels(level_idx)
        level_max_steps, level_sims = get_level_settings(current_level, args.max_steps, args.num_sims)

        # Cycle through all levels in this stage for map diversity
        play_level = stage_levels[iters_on_current % len(stage_levels)]
        levels_seen.add(play_level)

        if iters_on_current == 0:
            recent_win_rates.clear()
            print(f"\n{'#'*60}")
            print(f"  CURRICULUM: Level {current_level} "
                  f"(stage {level_idx + 1}/{len(levels)}, {len(stage_levels)} maps, "
                  f"max_steps={level_max_steps}, sims={level_sims})")
            print(f"{'#'*60}")

        print(f"\n{'='*60}")
        print(f"Iteration {iteration} (playing L{play_level}, attempt {iters_on_current + 1}/{args.iters_per_level})")
        print(f"{'='*60}")

        # --- Self-play on rotated level ---
        t0 = time.time()
        sp_model_path = best_model_onnx if os.path.exists(best_model_onnx) else None

        examples, sp_stats = run_self_play(
            level=play_level,
            num_games=args.games_per_iter,
            num_simulations=level_sims,
            c_puct=args.c_puct,
            action_repeat=args.action_repeat,
            max_steps=level_max_steps,
            model_path=sp_model_path,
        )

        data_dir = f"data/alphazero/{play_level}"
        iter_data_path = os.path.join(data_dir, f"iteration_{iteration}.npz")
        save_examples(examples, iter_data_path)
        print(f"  Self-play: {len(examples)} examples in {time.time() - t0:.1f}s")

        # --- Collect replay buffer from all seen levels ---
        all_examples = []
        start_iter = max(0, iteration - args.buffer_size + 1)
        for lvl in levels_seen:
            lvl_data_dir = f"data/alphazero/{lvl}"
            for i in range(start_iter, iteration + 1):
                path = os.path.join(lvl_data_dir, f"iteration_{i}.npz")
                if os.path.exists(path):
                    all_examples.extend(load_examples(path))
        print(f"  Replay buffer: {len(all_examples)} examples ({len(levels_seen)} levels, iters {start_iter}-{iteration})")

        # --- Train ---
        t0 = time.time()
        metrics = train_network(model, all_examples, optimizer, args.epochs, args.batch_size)
        print(f"  Loss: policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f} ({time.time() - t0:.1f}s)")

        # --- Export ---
        iter_onnx = os.path.join(save_dir, f"model_iter_{iteration}.onnx")
        iter_pt = os.path.join(save_dir, f"model_iter_{iteration}.pt")
        torch.save(model.state_dict(), iter_pt)
        export_to_onnx(model, iter_onnx)

        # --- Evaluate on the stage's hardest level (not the one we just played) ---
        eval_level = stage_levels[-1]
        new_results = evaluate_model(
            eval_level, iter_onnx, args.eval_games,
            num_sims=level_sims // 2,
            action_repeat=args.action_repeat,
            max_steps=level_max_steps,
        )
        win_rate = new_results["win_rate"]
        recent_win_rates.append(win_rate)
        smoothed = sum(recent_win_rates[-WIN_RATE_WINDOW:]) / min(len(recent_win_rates), WIN_RATE_WINDOW)
        print(f"  Eval (L{eval_level}): reward={new_results['mean_reward']:.1f}, "
              f"pickups={new_results['mean_pickups']:.1f}, "
              f"wins={new_results['wins']}/{new_results['total']} ({win_rate:.0%}), "
              f"smoothed={smoothed:.0%}")

        # --- Model promotion ---
        torch.save(model.state_dict(), best_model_pt)
        export_to_onnx(model, best_model_onnx)

        iters_on_current += 1
        iteration += 1

        # --- Auto-advancement (smoothed over last N evals) ---
        if len(recent_win_rates) >= WIN_RATE_WINDOW and smoothed >= args.win_threshold:
            print(f"  >>> Advanced! Smoothed win rate {smoothed:.0%} >= {args.win_threshold:.0%} "
                  f"(last {WIN_RATE_WINDOW}: {', '.join(f'{r:.0%}' for r in recent_win_rates[-WIN_RATE_WINDOW:])})")
            level_idx += 1
            iters_on_current = 0
        elif iters_on_current >= args.iters_per_level:
            print(f"  >>> Advancing (hit max {args.iters_per_level} iters, smoothed={smoothed:.0%})")
            level_idx += 1
            iters_on_current = 0

    if level_idx >= len(levels):
        print(f"\n{'='*60}")
        print(f"Curriculum complete! Passed all {len(levels)} levels.")
    else:
        print(f"\n{'='*60}")
        print(f"Training budget exhausted at level {levels[level_idx]} ({level_idx + 1}/{len(levels)})")
    print(f"Best model: {best_model_onnx}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
