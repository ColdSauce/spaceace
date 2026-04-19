"""AlphaZero training on a single level, optimizing for completion time.

Loop:
  1. Self-play N games with current model (heuristic fallback on iter 0).
  2. Train AlphaZeroNet on (obs, MCTS-policy, discounted-return) tuples.
  3. Export ONNX for Rust inference.
  4. Deterministic eval: run one greedy game, measure completion time.
  5. If faster than the current AI ghost, save it as the new ghost.

Self-play value targets are already time-aware in Rust:
  - Completion: 0.5 + 0.5 × (1 − step/max_steps) → faster = higher value
  - Crash: −1.0
  - Truncation: 0
  - Discounted backward through the game at γ=0.99

So the value head learns to predict "how quickly will this state complete," which
is exactly the signal we need to surpass a heuristic that can't see burn-flip-brake.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import spaceace_rl  # noqa: E402
from dashboard.db import get_db, init_db  # noqa: E402
from spaceace.agents.alphazero.network import AlphaZeroNet, export_to_onnx  # noqa: E402
from spaceace.agents.alphazero.self_play import run_self_play, GameExample  # noqa: E402
from spaceace.agents.alphazero.train import train_network  # noqa: E402


def bootstrap_from_vanilla_mcts(
    level: int, max_steps: int, num_games: int, num_simulations: int, action_repeat: int
) -> list[GameExample]:
    """Run vanilla (heuristic) MCTS to generate initial training examples.

    Each decision point produces (obs, visit-distribution policy, heuristic value)
    in the AlphaZero format. Uses the 27-dim alphazero observation so targets match.
    """
    from spaceace.core.env import SpaceAceDirectEnv

    examples: list[GameExample] = []
    az_engine = spaceace_rl.PyAlphaZeroEngine(level, max_steps, None)  # for obs + heuristic
    mcts_engine = spaceace_rl.PyMCTSEngine(level, max_steps, False)

    print(f"  [bootstrap] running {num_games} vanilla-MCTS games "
          f"({num_simulations} sims, ar={action_repeat})...", flush=True)

    for g in range(num_games):
        env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
        env.reset()
        game_examples: list[GameExample] = []
        step = 0
        pending_action_idx = None
        pending_repeats = 0
        completed = False
        crashed = False

        while step < max_steps:
            if pending_repeats > 0:
                pending_repeats -= 1
                action = _action_from_index(pending_action_idx)
            else:
                state = env.save_state()
                # Record AZ-format observation and MCTS visit distribution at this state
                obs = az_engine.get_observation(state)
                _, action_stats, _ = mcts_engine.search_with_stats(
                    state, num_simulations, action_repeat, 1.41, 0.99, 0.5
                )
                visits = np.zeros(6, dtype=np.float32)
                for a_idx, v, _mv in action_stats:
                    visits[a_idx] = float(v)
                if visits.sum() > 0:
                    # Apply temperature 0.5 to sharpen toward best (but not fully greedy)
                    sharpened = visits ** 2.0
                    policy = sharpened / sharpened.sum()
                else:
                    policy = np.ones(6, dtype=np.float32) / 6
                game_examples.append(GameExample(
                    observation=np.array(obs, dtype=np.float32),
                    mcts_policy=policy,
                ))

                # Sample action from policy (introduces diversity across games)
                best_action = int(np.argmax(policy))
                pending_action_idx = best_action
                pending_repeats = action_repeat - 1
                action = _action_from_index(pending_action_idx)

            obs_step, _, term, trunc, info = env.step(action)
            step += 1
            if info.get("level_completed"):
                completed = True
                break
            if info.get("ship_exploded"):
                crashed = True
                break
            if term or trunc:
                break

        # Assign time-discounted value targets (same formula as Rust play_games)
        if completed:
            time_remaining = 1.0 - step / max_steps
            outcome = 0.5 + 0.5 * time_remaining
        elif crashed:
            outcome = -1.0
        else:
            outcome = 0.0
        discount = 0.99
        n = len(game_examples)
        for i, ex in enumerate(game_examples):
            steps_from_end = n - 1 - i
            ex.value_target = float(outcome * (discount ** steps_from_end))

        examples.extend(game_examples)
        print(f"    game {g+1}/{num_games}: step={step} {'completed' if completed else 'crashed' if crashed else 'truncated'} "
              f"(examples so far: {len(examples)})", flush=True)
        env.close()

    return examples


_BOOTSTRAP_ACTIONS_CACHE = [
    np.array([0, 0, 0], dtype=np.int32),
    np.array([0, 0, 1], dtype=np.int32),
    np.array([1, 0, 0], dtype=np.int32),
    np.array([1, 0, 1], dtype=np.int32),
    np.array([0, 1, 0], dtype=np.int32),
    np.array([0, 1, 1], dtype=np.int32),
]


def _action_from_index(idx: int) -> np.ndarray:
    return _BOOTSTRAP_ACTIONS_CACHE[idx]

ALL_ACTIONS = [
    np.array([0, 0, 0], dtype=np.int32),
    np.array([0, 0, 1], dtype=np.int32),
    np.array([1, 0, 0], dtype=np.int32),
    np.array([1, 0, 1], dtype=np.int32),
    np.array([0, 1, 0], dtype=np.int32),
    np.array([0, 1, 1], dtype=np.int32),
]


def run_deterministic_game(
    level: int, max_steps: int, model_path: str | None,
    num_sims: int, action_repeat: int, c_puct: float,
) -> tuple[str, int, list]:
    """Greedy game with the current model. Returns (outcome, total_frames, per_frame_data)."""
    from spaceace.core.env import SpaceAceDirectEnv

    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    engine = spaceace_rl.PyAlphaZeroEngine(level, max_steps, model_path)
    env.reset()
    frames: list[dict] = []
    step = 0
    pending_action: np.ndarray | None = None
    pending_repeats = 0

    while step < max_steps:
        if pending_repeats > 0:
            action = pending_action
            pending_repeats -= 1
        else:
            state = env.save_state()
            action_idx, _, _ = engine.search(
                state, num_sims, c_puct,
                temperature=0.01, action_repeat=action_repeat,
                dirichlet_alpha=0.3, dirichlet_epsilon=0.0,
            )
            action = ALL_ACTIONS[int(action_idx)]
            pending_action = action
            pending_repeats = action_repeat - 1

        obs, _, term, trunc, info = env.step(action)
        step += 1
        frames.append({
            "x": round(float(obs[0]), 1),
            "y": round(float(obs[1]), 1),
            "rotation": round(float(obs[4]), 3),
            "thrusting": int(action[2]) > 0,
        })
        if info.get("level_completed"):
            env.close()
            return "completed", step, frames
        if info.get("ship_exploded"):
            env.close()
            return "crashed", step, frames
        if term or trunc:
            break

    env.close()
    return "truncated", step, frames


def save_ghost_if_better(level: int, frames: list, time_seconds: float) -> bool:
    """Save downsampled ghost trajectory to the DB if faster than existing AI ghost."""
    ghost_frames = []
    for i, f in enumerate(frames):
        if i % 6 == 0 or i == len(frames) - 1:
            ghost_frames.append({
                "x": f["x"], "y": f["y"],
                "rotation": f["rotation"],
                "thrusting": f["thrusting"],
                "time": round(i / 60.0, 3),
            })

    init_db()
    db = get_db()
    existing = db.execute(
        "SELECT time_seconds FROM ghost_replays WHERE level = ? AND ghost_type = 'ai'",
        (level,),
    ).fetchone()
    if existing and existing["time_seconds"] <= time_seconds:
        print(f"    existing AI ghost ({existing['time_seconds']:.2f}s) ≤ "
              f"new ({time_seconds:.2f}s), not saving", flush=True)
        db.close()
        return False
    db.execute(
        """INSERT OR REPLACE INTO ghost_replays
           (level, ghost_type, steps, time_seconds, frames_json)
           VALUES (?, 'ai', ?, ?, ?)""",
        (level, len(ghost_frames), time_seconds, json.dumps(ghost_frames)),
    )
    db.commit()
    db.close()
    print(f"    ★ NEW GHOST saved: {time_seconds:.2f}s", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--games-per-iter", type=int, default=64)
    p.add_argument("--num-sims", type=int, default=400)
    p.add_argument("--action-repeat", type=int, default=3)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--max-steps", type=int, default=2500)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--buffer-size", type=int, default=5,
                   help="Keep examples from the last N iterations as replay buffer")
    p.add_argument("--eval-sims", type=int, default=None,
                   help="MCTS sims at evaluation time (default: 2x training sims)")
    p.add_argument("--temp-threshold", type=int, default=10,
                   help="First N decisions use temperature=1.0 (exploration), rest use 0.1")
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.1,
                   help="Weight of Dirichlet noise on root priors during self-play")
    p.add_argument("--resume", action="store_true",
                   help="Load existing best_model.pt if it exists")
    p.add_argument("--bootstrap-games", type=int, default=4,
                   help="Run N vanilla-MCTS games to pretrain the network (0 to skip)")
    p.add_argument("--bootstrap-sims", type=int, default=20000,
                   help="MCTS sims per move during bootstrap")
    args = p.parse_args()

    save_dir = Path(f"models/alphazero/{args.level}")
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(f"data/alphazero/{args.level}")
    data_dir.mkdir(parents=True, exist_ok=True)

    best_model_pt = save_dir / "best_model.pt"
    best_model_onnx = save_dir / "best_model.onnx"

    net = AlphaZeroNet()
    if args.resume and best_model_pt.exists():
        net.load_state_dict(torch.load(best_model_pt, weights_only=True))
        print(f"Resumed from {best_model_pt}", flush=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    eval_sims = args.eval_sims or (args.num_sims * 2)

    print(f"=== AlphaZero Single-Level Training ===", flush=True)
    print(f"Level: {args.level}", flush=True)
    print(f"Iterations: {args.iterations}", flush=True)
    print(f"Games/iter: {args.games_per_iter}", flush=True)
    print(f"Train sims: {args.num_sims}, eval sims: {eval_sims}", flush=True)
    print(f"Action repeat: {args.action_repeat}, max_steps: {args.max_steps}", flush=True)
    print(f"Save dir: {save_dir}", flush=True)
    print(flush=True)

    buffer_paths: list[Path] = []
    best_time_seconds = float("inf")
    bootstrap_examples: list[GameExample] = []

    # --- Bootstrap: run vanilla MCTS a few times to generate policy distillation data ---
    if args.bootstrap_games > 0 and not best_model_pt.exists():
        print(f"=== Bootstrap phase ===", flush=True)
        t0 = time.time()
        bootstrap_examples = bootstrap_from_vanilla_mcts(
            level=args.level,
            max_steps=args.max_steps,
            num_games=args.bootstrap_games,
            num_simulations=args.bootstrap_sims,
            action_repeat=args.action_repeat,
        )
        print(f"  [bootstrap] collected {len(bootstrap_examples)} examples "
              f"({time.time()-t0:.1f}s)", flush=True)

        # Pretrain network on bootstrap data
        print(f"  [pretrain] training on bootstrap examples...", flush=True)
        t0 = time.time()
        metrics = train_network(net, bootstrap_examples, optimizer,
                                epochs=args.epochs * 2, batch_size=args.batch_size)
        print(f"  [pretrain] policy_loss={metrics['policy_loss']:.4f} "
              f"value_loss={metrics['value_loss']:.4f} ({time.time()-t0:.1f}s)", flush=True)
        torch.save(net.state_dict(), str(best_model_pt))
        export_to_onnx(net, str(best_model_onnx))

        # Save bootstrap shard so it stays in buffer for a few iters
        shard_path = data_dir / "bootstrap.npz"
        from spaceace.agents.alphazero.self_play import save_examples
        save_examples(bootstrap_examples, str(shard_path))
        buffer_paths.append(shard_path)

        # Evaluate pretrained net
        outcome, eval_steps, eval_frames = run_deterministic_game(
            level=args.level,
            max_steps=args.max_steps,
            model_path=str(best_model_onnx),
            num_sims=eval_sims,
            action_repeat=args.action_repeat,
            c_puct=args.c_puct,
        )
        if outcome == "completed":
            eval_seconds = eval_steps / 60.0
            print(f"  [bootstrap eval] ✓ completed {eval_seconds:.2f}s", flush=True)
            if eval_seconds < best_time_seconds:
                if save_ghost_if_better(args.level, eval_frames, eval_seconds):
                    best_time_seconds = eval_seconds
        else:
            print(f"  [bootstrap eval] ✗ {outcome} at {eval_steps} frames", flush=True)
        print(flush=True)

    # Check current AI ghost to know what to beat
    init_db()
    db = get_db()
    row = db.execute(
        "SELECT time_seconds FROM ghost_replays WHERE level = ? AND ghost_type = 'ai'",
        (args.level,),
    ).fetchone()
    db.close()
    if row:
        best_time_seconds = row["time_seconds"]
        print(f"Current AI ghost: {best_time_seconds:.2f}s — target to beat", flush=True)
    else:
        print("No current AI ghost for this level", flush=True)
    print(flush=True)

    for iteration in range(args.iterations):
        print(f"=== Iteration {iteration} ===", flush=True)

        # --- Self-play ---
        t0 = time.time()
        model_path = str(best_model_onnx) if best_model_onnx.exists() else None
        mode = "NN" if model_path else "heuristic"
        print(f"  [self-play] {args.games_per_iter} games with {mode} model...", flush=True)

        examples, sp_stats = run_self_play(
            level=args.level,
            num_games=args.games_per_iter,
            num_simulations=args.num_sims,
            c_puct=args.c_puct,
            action_repeat=args.action_repeat,
            max_steps=args.max_steps,
            model_path=model_path,
            temp_threshold=args.temp_threshold,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon,
        )
        sp_elapsed = time.time() - t0
        wins = sum(1 for s in sp_stats if s.completed)
        completion_times = [s.steps for s in sp_stats if s.completed]
        if completion_times:
            best_sp = min(completion_times)
            avg_sp = sum(completion_times) / len(completion_times)
            print(f"  [self-play] {len(examples)} examples, {wins}/{len(sp_stats)} wins, "
                  f"best={best_sp/60:.2f}s avg={avg_sp/60:.2f}s ({sp_elapsed:.1f}s)", flush=True)
        else:
            print(f"  [self-play] {len(examples)} examples, {wins}/{len(sp_stats)} wins "
                  f"({sp_elapsed:.1f}s)", flush=True)

        # Save shard to disk, update buffer
        shard_path = data_dir / f"iter_{iteration}.npz"
        from spaceace.agents.alphazero.self_play import save_examples, load_examples
        save_examples(examples, str(shard_path))
        buffer_paths.append(shard_path)
        if len(buffer_paths) > args.buffer_size:
            old = buffer_paths.pop(0)
            # Keep on disk but don't load into buffer

        # --- Collect replay buffer ---
        all_examples: list[GameExample] = []
        for p in buffer_paths:
            all_examples.extend(load_examples(str(p)))
        print(f"  [buffer] {len(all_examples)} examples from {len(buffer_paths)} iters", flush=True)

        # --- Train network ---
        t0 = time.time()
        metrics = train_network(net, all_examples, optimizer,
                                epochs=args.epochs, batch_size=args.batch_size)
        print(f"  [train] policy_loss={metrics['policy_loss']:.4f} "
              f"value_loss={metrics['value_loss']:.4f} ({time.time()-t0:.1f}s)", flush=True)

        # --- Export ---
        torch.save(net.state_dict(), str(best_model_pt))
        export_to_onnx(net, str(best_model_onnx))

        # --- Deterministic eval ---
        t0 = time.time()
        outcome, eval_steps, eval_frames = run_deterministic_game(
            level=args.level,
            max_steps=args.max_steps,
            model_path=str(best_model_onnx),
            num_sims=eval_sims,
            action_repeat=args.action_repeat,
            c_puct=args.c_puct,
        )
        eval_elapsed = time.time() - t0
        if outcome == "completed":
            eval_seconds = eval_steps / 60.0
            print(f"  [eval] ✓ completed {eval_steps} frames = {eval_seconds:.2f}s "
                  f"(best_so_far={best_time_seconds:.2f}s, {eval_elapsed:.1f}s)", flush=True)
            if eval_seconds < best_time_seconds:
                if save_ghost_if_better(args.level, eval_frames, eval_seconds):
                    best_time_seconds = eval_seconds
        else:
            print(f"  [eval] ✗ {outcome} at {eval_steps} frames ({eval_elapsed:.1f}s)", flush=True)

        print(flush=True)

    print(f"=== Training complete ===", flush=True)
    print(f"Best eval time: {best_time_seconds:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
