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
    evaluate_model,
    generate_curriculum_maps,
    train_network,
)
from spaceace.training.trainer import LevelStage, Trainer, TrainingConfig


class AlphaZeroTrainer(Trainer):
    """AlphaZero training: self-play via Rust MCTS, supervised network training.

    Does not use SB3 or gym VecEnvs. The training loop is:
      1. Self-play on the current curriculum stage (Rust ``PyAlphaZeroEngine``)
      2. Collect replay buffer from recent iterations across seen levels
      3. Train ``AlphaZeroNet`` (dual-head: policy + value) on collected examples
      4. Export to ONNX for Rust inference
      5. Evaluate on the hardest level in the current stage
      6. Advance when smoothed gate completion >= stage.advance_win_rate AND
         iters_on_stage >= stage.min_iters (hard cap: az.iters_per_level)
    """

    def fit(self, config: TrainingConfig) -> Path:
        az = config.alphazero
        if az.advance_metric not in {"self_play", "eval"}:
            raise ValueError(
                f"Unsupported AlphaZero advance_metric={az.advance_metric!r}; "
                "expected 'self_play' or 'eval'"
            )

        if az.generate_curriculum:
            generate_curriculum_maps()

        # Normalize inputs to a list of LevelStages.
        if config.curriculum is not None and len(config.curriculum) > 0:
            stages = list(config.curriculum)
        elif config.level is not None:
            stages = [LevelStage(levels=[config.level])]
        else:
            from spaceace.training.curriculum import build_curriculum
            stages = build_curriculum(list(range(3000, 3165)))

        all_levels = sorted({lvl for s in stages for lvl in s.levels})

        save_dir = config.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        if az.fresh:
            import shutil
            for lvl in all_levels:
                data_dir = Path(f"data/alphazero/{lvl}")
                if data_dir.exists():
                    shutil.rmtree(data_dir)
            if save_dir.exists():
                shutil.rmtree(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
            print("Cleared previous training data and models.")

        print(f"=== AlphaZero Training ===")
        print(f"Curriculum: {len(stages)} stages across {len(all_levels)} unique levels")
        print(f"Total iteration budget: {az.iterations}")
        print(f"Games/iter: {az.games_per_iteration}, Sims/move: {az.simulations_per_move}")
        print(f"Hard cap per stage: {az.iters_per_level} iterations")
        print()

        # Initialize or load model.
        net = AlphaZeroNet()
        best_model_pt = save_dir / "best_model.pt"
        best_model_onnx = save_dir / "best_model.onnx"
        best_model_available = False

        resume_iter = 0
        if config.resume_from and best_model_pt.exists():
            net.load_state_dict(torch.load(best_model_pt, weights_only=True))
            best_model_available = best_model_onnx.exists()
            try:
                resume_iter = int(config.resume_from)
            except (TypeError, ValueError):
                resume_iter = 0
            print(f"Resumed from {best_model_pt} at iteration {resume_iter}")

        optimizer = torch.optim.Adam(net.parameters(), lr=az.network_lr, weight_decay=1e-4)

        for lvl in all_levels:
            os.makedirs(f"data/alphazero/{lvl}", exist_ok=True)

        iteration = resume_iter
        end_iteration = resume_iter + az.iterations
        levels_seen: set[int] = set()
        generated_shards: dict[tuple[int, int], str] = {}

        budget_exhausted = False
        last_stage_idx = -1

        for stage_idx, stage in enumerate(stages):
            if iteration >= end_iteration:
                budget_exhausted = True
                last_stage_idx = stage_idx
                break

            last_stage_idx = stage_idx
            stage_levels = list(stage.levels)
            stage_max_steps = stage.max_episode_steps or config.max_episode_steps
            stage_sims = az.simulations_per_move
            stage_win_target = stage.advance_win_rate
            stage_min_iters = max(1, stage.min_iters)
            hard_cap = max(stage_min_iters, az.iters_per_level)

            print(f"\n{'#' * 60}")
            print(
                f"  STAGE {stage_idx + 1}/{len(stages)}: levels={stage_levels} "
                f"(max_steps={stage_max_steps}, sims={stage_sims})"
            )
            print(
                f"  Advance target: {az.advance_metric} completion >= {stage_win_target:.0%} "
                f"after >= {stage_min_iters} iters (hard cap {hard_cap})"
            )
            print(f"{'#' * 60}")

            recent_gate_rates: list[float] = []
            iters_on_stage = 0
            advanced = False
            # Per-stage best tracking: harder stages shouldn't inherit the
            # easier stage's score, or no candidate would ever get promoted.
            best_score = None

            while iters_on_stage < hard_cap and iteration < end_iteration:
                play_level = stage_levels[iters_on_stage % len(stage_levels)]
                levels_seen.add(play_level)

                print(f"\n{'=' * 60}")
                print(
                    f"Iteration {iteration} (stage {stage_idx + 1}, "
                    f"L{play_level}, attempt {iters_on_stage + 1}/{hard_cap})"
                )
                print(f"{'=' * 60}")

                # --- Self-play ---
                t0 = time.time()
                sp_model_path = str(best_model_onnx) if best_model_available else None

                examples, _sp_stats = run_self_play(
                    level=play_level,
                    num_games=az.games_per_iteration,
                    num_simulations=stage_sims,
                    c_puct=az.c_puct,
                    action_repeat=config.action_repeat,
                    max_steps=stage_max_steps,
                    model_path=sp_model_path,
                )
                self_play_total = len(_sp_stats)
                self_play_wins = sum(1 for s in _sp_stats if s.completed)
                self_play_win_rate = self_play_wins / max(self_play_total, 1)

                iter_data_path = os.path.join(
                    f"data/alphazero/{play_level}", f"iteration_{iteration}.npz"
                )
                save_examples(examples, iter_data_path)
                generated_shards[(play_level, iteration)] = iter_data_path
                print(f"  Self-play: {len(examples)} examples in {time.time() - t0:.1f}s")

                # --- Replay buffer ---
                all_examples: list[GameExample] = []
                start_iter = max(0, iteration - az.replay_buffer_shards + 1)
                for lvl in levels_seen:
                    lvl_data_dir = f"data/alphazero/{lvl}"
                    for i in range(start_iter, iteration + 1):
                        path = generated_shards.get((lvl, i))
                        if path is None and config.resume_from:
                            path = os.path.join(lvl_data_dir, f"iteration_{i}.npz")
                        if path is not None and os.path.exists(path):
                            all_examples.extend(load_examples(path))
                print(
                    f"  Replay buffer: {len(all_examples)} examples "
                    f"({len(levels_seen)} levels, iters {start_iter}-{iteration})"
                )

                # --- Train network ---
                t0 = time.time()
                metrics = train_network(
                    net, all_examples, optimizer,
                    az.network_train_epochs, az.network_batch_size,
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

                # --- Evaluate on hardest level in this stage ---
                eval_level = stage_levels[-1]
                results = evaluate_model(
                    eval_level,
                    iter_onnx,
                    az.eval_games,
                    num_sims=max(1, stage_sims // 2),
                    action_repeat=config.action_repeat,
                    max_steps=stage_max_steps,
                )
                eval_win_rate = results["win_rate"]
                gate_rate = (
                    self_play_win_rate
                    if az.advance_metric == "self_play"
                    else eval_win_rate
                )
                recent_gate_rates.append(gate_rate)
                window = max(1, az.win_rate_window)
                smoothed = sum(recent_gate_rates[-window:]) / min(
                    len(recent_gate_rates), window
                )
                print(
                    f"  Eval (L{eval_level}): reward={results['mean_reward']:.1f}, "
                    f"pickups={results['mean_pickups']:.1f}, "
                    f"wins={results['wins']}/{results['total']} ({eval_win_rate:.1%}), "
                    f"smoothed={smoothed:.1%}, gate={gate_rate:.1%} ({az.advance_metric})"
                )

                # --- Promote best model (only if eval improved) ---
                candidate_score = (eval_win_rate, float(results["mean_reward"]))
                if best_score is None or candidate_score > best_score:
                    torch.save(net.state_dict(), str(best_model_pt))
                    export_to_onnx(net, str(best_model_onnx))
                    best_model_available = True
                    best_score = candidate_score
                    print(
                        f"  -> promoted to best (win_rate={eval_win_rate:.0%}, "
                        f"reward={candidate_score[1]:.1f})"
                    )
                else:
                    print(
                        f"  -> kept prior best (win_rate={best_score[0]:.0%}, "
                        f"reward={best_score[1]:.1f}); "
                        f"candidate (win_rate={candidate_score[0]:.0%}, reward={candidate_score[1]:.1f})"
                    )

                iters_on_stage += 1
                iteration += 1

                # --- Stage advancement check ---
                if (
                    iters_on_stage >= stage_min_iters
                    and smoothed >= stage_win_target
                ):
                    print(
                        f"  >>> Advancing stage ({az.advance_metric} smoothed {smoothed:.0%} >= "
                        f"{stage_win_target:.0%} after {iters_on_stage} iters)"
                    )
                    advanced = True
                    break

            if not advanced and iteration < end_iteration:
                window = max(1, az.win_rate_window)
                recent_smoothed = sum(recent_gate_rates[-window:]) / max(
                    1, min(len(recent_gate_rates), window)
                )
                print(
                    f"  >>> Force-advancing stage (hit hard cap {hard_cap}, "
                    f"{az.advance_metric} smoothed={recent_smoothed:.0%})"
                )

            if iteration >= end_iteration:
                budget_exhausted = True
                break

        print(f"\n{'=' * 60}")
        if budget_exhausted:
            print(
                f"Training budget exhausted at stage "
                f"{last_stage_idx + 1}/{len(stages)}"
            )
        else:
            print(f"Curriculum complete! Passed all {len(stages)} stages.")
        print(f"Best model: {best_model_onnx}")

        return best_model_pt
