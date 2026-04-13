"""AlphaZero Trainer subclass: self-play -> train network -> evaluate -> repeat."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from spaceace.agents.alphazero.network import AlphaZeroNet, export_to_onnx
from spaceace.agents.alphazero.self_play import (
    GameExample,
    load_examples,
    run_self_play,
    save_examples,
)
from spaceace.agents.alphazero.train import (
    CURRICULUM_STAGES,
    evaluate_model,
    generate_curriculum_maps,
    get_level_settings,
    train_network,
)
from spaceace.training.trainer import Trainer, TrainingConfig


class AlphaZeroTrainer(Trainer):
    """AlphaZero training: self-play via Rust MCTS, supervised network training.

    Does not use SB3 or gym VecEnvs. The training loop is:
      1. Self-play on current curriculum level (Rust ``PyAlphaZeroEngine``)
      2. Collect replay buffer from recent iterations across seen levels
      3. Train ``AlphaZeroNet`` (dual-head: policy + value) on collected examples
      4. Export to ONNX for Rust inference
      5. Evaluate on hardest level in current stage
      6. Auto-advance based on smoothed win rate
    """

    def fit(self, config: TrainingConfig) -> Path:
        az = config.alphazero

        if az.generate_curriculum:
            generate_curriculum_maps()

        # Determine level schedule
        if config.curriculum is not None:
            levels = []
            for stage in config.curriculum:
                levels.extend(stage.levels)
        elif config.level is not None:
            levels = [config.level]
        else:
            from spaceace.agents.alphazero.train import DEFAULT_CURRICULUM
            levels = DEFAULT_CURRICULUM

        save_dir = config.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        if az.fresh:
            import shutil
            for lvl in levels:
                data_dir = Path(f"data/alphazero/{lvl}")
                if data_dir.exists():
                    shutil.rmtree(data_dir)
            if save_dir.exists():
                shutil.rmtree(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
            print("Cleared previous training data and models.")

        print(f"=== AlphaZero Training ===")
        print(f"Curriculum: {len(levels)} levels (L{levels[0]} -> L{levels[-1]})")
        print(f"Win threshold to advance: {az.win_threshold:.0%}")
        print(f"Max iters/level: {az.iters_per_level}")
        print(f"Total iterations: {az.iterations}")
        print(f"Games/iter: {az.games_per_iteration}")
        print(f"Sims/move: {az.simulations_per_move}")
        print()

        # Initialize or load model
        net = AlphaZeroNet()
        best_model_pt = save_dir / "best_model.pt"
        best_model_onnx = save_dir / "best_model.onnx"

        resume_iter = 0
        if config.resume_from and best_model_pt.exists():
            net.load_state_dict(torch.load(best_model_pt, weights_only=True))
            resume_iter = int(config.resume_from)
            print(f"Resumed from {best_model_pt} at iteration {resume_iter}")

        optimizer = torch.optim.Adam(net.parameters(), lr=az.network_lr, weight_decay=1e-4)

        iteration = resume_iter
        end_iteration = resume_iter + az.iterations
        level_idx = 0
        iters_on_current = 0
        levels_seen: set[int] = set()
        recent_win_rates: list[float] = []

        def _get_stage_levels(lvl_idx: int) -> list[int]:
            target = levels[lvl_idx]
            for level_range, _, _, _ in CURRICULUM_STAGES:
                if target in level_range:
                    return [l for l in levels if l in level_range]
            return [target]

        # Ensure data dirs exist
        for lvl in levels:
            os.makedirs(f"data/alphazero/{lvl}", exist_ok=True)

        while iteration < end_iteration and level_idx < len(levels):
            current_level = levels[level_idx]
            stage_levels = _get_stage_levels(level_idx)
            level_max_steps, level_sims = get_level_settings(
                current_level, config.max_episode_steps, az.simulations_per_move
            )

            play_level = stage_levels[iters_on_current % len(stage_levels)]
            levels_seen.add(play_level)

            if iters_on_current == 0:
                recent_win_rates.clear()
                print(f"\n{'#' * 60}")
                print(
                    f"  CURRICULUM: Level {current_level} "
                    f"(stage {level_idx + 1}/{len(levels)}, {len(stage_levels)} maps, "
                    f"max_steps={level_max_steps}, sims={level_sims})"
                )
                print(f"{'#' * 60}")

            print(f"\n{'=' * 60}")
            print(
                f"Iteration {iteration} (playing L{play_level}, "
                f"attempt {iters_on_current + 1}/{az.iters_per_level})"
            )
            print(f"{'=' * 60}")

            # --- Self-play ---
            t0 = time.time()
            sp_model_path = str(best_model_onnx) if best_model_onnx.exists() else None

            examples, sp_stats = run_self_play(
                level=play_level,
                num_games=az.games_per_iteration,
                num_simulations=level_sims,
                c_puct=az.c_puct,
                action_repeat=config.action_repeat,
                max_steps=level_max_steps,
                model_path=sp_model_path,
            )

            data_dir = f"data/alphazero/{play_level}"
            iter_data_path = os.path.join(data_dir, f"iteration_{iteration}.npz")
            save_examples(examples, iter_data_path)
            print(f"  Self-play: {len(examples)} examples in {time.time() - t0:.1f}s")

            # --- Collect replay buffer ---
            all_examples: list[GameExample] = []
            start_iter = max(0, iteration - az.replay_buffer_shards + 1)
            for lvl in levels_seen:
                lvl_data_dir = f"data/alphazero/{lvl}"
                for i in range(start_iter, iteration + 1):
                    path = os.path.join(lvl_data_dir, f"iteration_{i}.npz")
                    if os.path.exists(path):
                        all_examples.extend(load_examples(path))
            print(
                f"  Replay buffer: {len(all_examples)} examples "
                f"({len(levels_seen)} levels, iters {start_iter}-{iteration})"
            )

            # --- Train network ---
            t0 = time.time()
            metrics = train_network(
                net, all_examples, optimizer, az.network_train_epochs, az.network_batch_size
            )
            print(
                f"  Loss: policy={metrics['policy_loss']:.4f} "
                f"value={metrics['value_loss']:.4f} ({time.time() - t0:.1f}s)"
            )

            # --- Export checkpoints ---
            iter_onnx = str(save_dir / f"model_iter_{iteration}.onnx")
            iter_pt = str(save_dir / f"model_iter_{iteration}.pt")
            torch.save(net.state_dict(), iter_pt)
            export_to_onnx(net, iter_onnx)

            # --- Evaluate on hardest level in stage ---
            eval_level = stage_levels[-1]
            new_results = evaluate_model(
                eval_level,
                iter_onnx,
                az.eval_games,
                num_sims=level_sims // 2,
                action_repeat=config.action_repeat,
                max_steps=level_max_steps,
            )
            win_rate = new_results["win_rate"]
            recent_win_rates.append(win_rate)
            smoothed = sum(recent_win_rates[-az.win_rate_window :]) / min(
                len(recent_win_rates), az.win_rate_window
            )
            print(
                f"  Eval (L{eval_level}): reward={new_results['mean_reward']:.1f}, "
                f"pickups={new_results['mean_pickups']:.1f}, "
                f"wins={new_results['wins']}/{new_results['total']} ({win_rate:.0%}), "
                f"smoothed={smoothed:.0%}"
            )

            # --- Model promotion ---
            torch.save(net.state_dict(), str(best_model_pt))
            export_to_onnx(net, str(best_model_onnx))

            iters_on_current += 1
            iteration += 1

            # --- Auto-advancement ---
            if len(recent_win_rates) >= az.win_rate_window and smoothed >= az.win_threshold:
                print(
                    f"  >>> Advanced! Smoothed win rate {smoothed:.0%} >= {az.win_threshold:.0%} "
                    f"(last {az.win_rate_window}: "
                    f"{', '.join(f'{r:.0%}' for r in recent_win_rates[-az.win_rate_window:])})"
                )
                level_idx += 1
                iters_on_current = 0
            elif iters_on_current >= az.iters_per_level:
                print(
                    f"  >>> Advancing (hit max {az.iters_per_level} iters, smoothed={smoothed:.0%})"
                )
                level_idx += 1
                iters_on_current = 0

        if level_idx >= len(levels):
            print(f"\n{'=' * 60}")
            print(f"Curriculum complete! Passed all {len(levels)} levels.")
        else:
            print(f"\n{'=' * 60}")
            print(
                f"Training budget exhausted at level {levels[level_idx]} "
                f"({level_idx + 1}/{len(levels)})"
            )
        print(f"Best model: {best_model_onnx}")

        return best_model_pt
