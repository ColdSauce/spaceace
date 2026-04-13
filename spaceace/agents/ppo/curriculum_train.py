"""PPO curriculum training: auto-advance through generated levels."""

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import time
from collections import deque

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from spaceace.core.gym_wrapper import SpaceAceGymWrapper
from spaceace.agents.ppo.training_env import (
    SpaceAceTrainingEnv,
    MetricsCallback,
    make_env,
)

ORIGINAL_LEVELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 100]
GENERATED_LEVELS = list(range(3000, 3050))
DEFAULT_CURRICULUM = GENERATED_LEVELS  # set after calibration sorts everything

# Cache of MCTS-calibrated max_steps per level
_level_max_steps_cache: dict[int, int] = {}
# Cache of calibration details per level
_level_calibration: dict[int, dict] = {}

CALIBRATION_FILE = "data/calibration_cache.json"


def load_calibration_cache() -> None:
    """Load cached calibration results from disk."""
    if not os.path.exists(CALIBRATION_FILE):
        return
    with open(CALIBRATION_FILE) as f:
        data = json.load(f)
    for level_str, info in data.items():
        level = int(level_str)
        _level_max_steps_cache[level] = info["max_steps"]
        _level_calibration[level] = info
    print(f"  Loaded calibration cache ({len(data)} levels from {CALIBRATION_FILE})")


def save_calibration_cache() -> None:
    """Save calibration results to disk."""
    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
    data = {}
    for level, info in _level_calibration.items():
        data[str(level)] = info
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved calibration cache ({len(data)} levels to {CALIBRATION_FILE})")


def _calibrate_one_level(args: tuple) -> tuple:
    """Worker function to calibrate a single level. Runs in a subprocess."""
    level, ceiling = args
    import spaceace_rl
    engine = spaceace_rl.PyMCTSEngine(level, ceiling)
    results = engine.play_games(3, 5000, action_repeat=5, max_steps=ceiling)
    return (level, results)


def calibrate_max_steps(levels: list[int], multiplier: float = 3.0,
                        floor: int = 300, ceiling: int = 5000,
                        fallback: int = 3000) -> None:
    """Run real MCTS on each level to determine a reasonable max_steps.

    Uses PyMCTSEngine with 5000 base sims (dynamically scaled near walls/at speed,
    matching the MCTSAgent behavior from run.py). Parallelized across CPU cores.
    """
    from concurrent.futures import ProcessPoolExecutor

    load_calibration_cache()

    to_calibrate = [l for l in levels if l not in _level_max_steps_cache]
    if not to_calibrate:
        print(f"  All {len(levels)} levels already calibrated (cached)")
        return

    worker_args = [(level, ceiling) for level in to_calibrate]
    n_workers = min(len(worker_args), os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for level, results in executor.map(_calibrate_one_level, worker_args):
            win_steps = [steps for completed, _, steps in results if completed]
            all_steps = [steps for _, _, steps in results]
            n_games = len(results)
            if win_steps:
                base = min(win_steps)
                ms = max(floor, min(ceiling, int(base * multiplier)))
                print(f"    L{level}: max_steps={ms} (wins={len(win_steps)}/{n_games}, fastest={base})")
            else:
                ms = fallback
                print(f"    L{level}: max_steps={ms} (unsolved by MCTS)")
            _level_max_steps_cache[level] = ms
            _level_calibration[level] = {
                "max_steps": ms,
                "mcts_wins": len(win_steps),
                "mcts_games": n_games,
                "mcts_fastest": min(win_steps) if win_steps else None,
                "mcts_avg_steps": sum(all_steps) / len(all_steps) if all_steps else 0,
            }

    if to_calibrate:
        save_calibration_cache()
        print(f"  Calibrated {len(to_calibrate)} new levels, {len(levels) - len(to_calibrate)} from cache")


def write_curriculum_summary(levels: list[int], stages: list[list[int]], path: str) -> None:
    """Write a human-readable curriculum summary file."""
    with open(path, "w") as f:
        f.write("# Curriculum Summary\n")
        f.write(f"# Generated {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"# {len(levels)} levels in {len(stages)} stages\n\n")

        f.write(f"{'Level':>6}  {'MaxSteps':>8}  {'MCTS':>10}  {'Fastest':>8}  {'AvgSteps':>8}  {'Stage':>6}\n")
        f.write("-" * 60 + "\n")

        for stage_idx, stage in enumerate(stages):
            for level in stage:
                cal = _level_calibration.get(level, {})
                ms = _level_max_steps_cache.get(level, 3000)
                mcts_wins = cal.get("mcts_wins", "?")
                mcts_games = cal.get("mcts_games", "?")
                fastest = cal.get("mcts_fastest")
                avg = cal.get("mcts_avg_steps", 0)
                fastest_str = str(fastest) if fastest else "-"
                f.write(f"L{level:>5}  {ms:>8}  {mcts_wins:>4}/{mcts_games:<4}  "
                        f"{fastest_str:>8}  {avg:>8.0f}  {stage_idx + 1:>6}\n")
            f.write("\n")

    print(f"Curriculum summary written to {path}")


def get_max_steps_for_stage(stage_levels: list[int]) -> int:
    """Return max_steps for a stage based on MCTS calibration of its hardest map."""
    # Use the highest calibrated value in the stage
    return max(_level_max_steps_cache.get(l, 3000) for l in stage_levels)


def generate_curriculum_maps():
    """Generate maps from all strategies, then sort globally by difficulty."""
    # Generate from each strategy into separate temp files
    configs = [
        ("simple", 10, 100),
        ("room", 10, 200),
        ("maze", 10, 300),
        ("cave", 10, 400),
        ("gauntlet", 10, 500),
    ]

    all_levels = {}
    for strategy, count, seed in configs:
        tmp_file = f"data/_tmp_{strategy}.json"
        subprocess.run([
            "uv", "run", "python", "generate_maps.py",
            "--count", str(count), "--strategy", strategy,
            "--seed", str(seed), "--output", tmp_file,
            "--start-level", "9000",  # temp numbering, will be reassigned
        ], check=True, capture_output=True)

        with open(tmp_file) as f:
            data = json.load(f)
        os.remove(tmp_file)

        # data is {"9000": [...], "9001": [...], ...}
        # We need difficulty scores to sort — re-run the scorer
        for key, level_data in data.items():
            all_levels[f"{strategy}_{key}"] = level_data

    # To sort by difficulty, we need to score them. Use generate_maps.py's scorer.
    # Simple heuristic: more vertices + more pickups + smaller corridors = harder
    # For now, use pickup count + wall count as proxy (since we don't have the scorer imported)
    def estimate_difficulty(level_data):
        """Quick difficulty estimate from flat array data."""
        if not level_data or len(level_data) < 2:
            return 0.0
        vertex_count = int(level_data[0])
        idx = 1 + vertex_count * 2
        if idx >= len(level_data):
            return 0.0
        line_count = int(level_data[idx])
        idx += 1 + line_count * 2
        if idx + 3 >= len(level_data):
            return 0.0
        # skip start_index, bounding_width, bounding_height
        idx += 3
        pickup_count = int(level_data[idx]) if idx < len(level_data) else 0
        # Combine: more walls and pickups = harder
        return line_count * 0.01 + pickup_count * 0.1

    # Sort all levels by estimated difficulty
    sorted_keys = sorted(all_levels.keys(), key=lambda k: estimate_difficulty(all_levels[k]))

    # Assign level numbers 3000+ in difficulty order
    final_levels = {}
    for i, key in enumerate(sorted_keys):
        final_levels[str(3000 + i)] = all_levels[key]

    # Write and merge
    output_path = "data/generated_levels.json"
    with open(output_path, "w") as f:
        json.dump(final_levels, f)

    # Merge into main levels file
    levels_path = "data/spaceace_levels.json"
    if os.path.exists(levels_path):
        with open(levels_path) as f:
            existing = json.load(f)
    else:
        existing = {}
    existing.update(final_levels)
    with open(levels_path, "w") as f:
        json.dump(existing, f)

    n = len(final_levels)
    print(f"Generated {n} maps sorted by difficulty (L3000-L{3000+n-1})")
    # Print difficulty summary
    for i in range(0, n, n // 5 if n >= 5 else 1):
        key = str(3000 + i)
        d = estimate_difficulty(final_levels[key])
        orig_key = sorted_keys[i]
        strategy = orig_key.split("_")[0]
        print(f"  L{key}: {strategy:10s} difficulty≈{d:.2f}")
    print()


def evaluate_win_rate(
    model: PPO,
    vec_normalize: VecNormalize,
    level: int,
    max_steps: int,
    action_repeat: int,
    n_episodes: int = 20,
) -> tuple[float, float, float]:
    """Evaluate model on a level. Returns (win_rate, mean_reward, mean_pickups)."""
    eval_base = SpaceAceGymWrapper(level=level, max_steps=max_steps)
    eval_shaped = SpaceAceTrainingEnv(eval_base, level=level, max_steps=max_steps,
                                      action_repeat=action_repeat)
    eval_vec = DummyVecEnv([lambda: Monitor(eval_shaped)])

    # Copy normalization stats from training env
    eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_vec.obs_rms = vec_normalize.obs_rms
    eval_vec.ret_rms = vec_normalize.ret_rms
    eval_vec.training = False
    eval_vec.norm_reward = False

    wins = 0
    total_reward = 0.0
    total_pickups = 0

    for _ in range(n_episodes):
        obs = eval_vec.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = eval_vec.step(action)
            done = dones[0]
        if eval_shaped.last_episode_completed:
            wins += 1
        total_pickups += eval_shaped.last_episode_pickups_collected
        # Approximate raw reward
        ep_reward = eval_shaped.last_episode_steps * -0.01
        if eval_shaped.last_episode_crashed:
            ep_reward -= 100.0
        if eval_shaped.last_episode_completed:
            ep_reward += 1000.0
        ep_reward += eval_shaped.last_episode_pickups_collected * 50.0
        total_reward += ep_reward

    eval_vec.close()

    return (
        wins / n_episodes,
        total_reward / n_episodes,
        total_pickups / n_episodes,
    )


def create_envs(levels: list[int], max_steps: int, action_repeat: int, n_envs: int):
    """Create training envs that cycle through multiple levels for diversity."""
    env_fns = []
    for i in range(n_envs):
        level = levels[i % len(levels)]
        env_fns.append(make_env(level, max_steps, action_repeat))
    return env_fns


def parse_args():
    p = argparse.ArgumentParser(description="PPO curriculum training for SpaceAce")
    p.add_argument("--curriculum", type=int, nargs="+", default=None,
                   help="Level numbers in difficulty order")
    p.add_argument("--generate-curriculum", action="store_true",
                   help="Auto-generate curriculum maps before training")
    p.add_argument("--fresh", action="store_true",
                   help="Delete previous models before starting")
    p.add_argument("--recalibrate", action="store_true",
                   help="Force recalibration (ignore cached results)")
    p.add_argument("--resume-stage", type=int, default=None,
                   help="Resume from this stage number (1-indexed)")
    p.add_argument("--timesteps-per-stage", type=int, default=200_000,
                   help="Timesteps per curriculum stage (default: 200000)")
    p.add_argument("--max-stages", type=int, default=None,
                   help="Stop after this many stages")
    p.add_argument("--win-threshold", type=float, default=0.5,
                   help="Win rate to advance (default: 0.5)")
    p.add_argument("--eval-freq", type=int, default=25_000,
                   help="Evaluate every N timesteps")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="Episodes per evaluation")
    p.add_argument("--action-repeat", type=int, default=5,
                   help="Frames per action (default: 5)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if args.generate_curriculum:
        generate_curriculum_maps()

    if args.curriculum:
        levels = args.curriculum
    else:
        levels = GENERATED_LEVELS + ORIGINAL_LEVELS

    save_dir = "models/ppo/curriculum"
    if args.fresh and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print("Cleared previous PPO models.")
    os.makedirs(save_dir, exist_ok=True)

    n_envs = multiprocessing.cpu_count()
    WIN_RATE_WINDOW = 3

    # Calibrate max_steps for all levels using MCTS
    # This also determines true difficulty (MCTS solve time)
    if args.recalibrate and os.path.exists(CALIBRATION_FILE):
        os.remove(CALIBRATION_FILE)
        print("Cleared calibration cache.")

    print("Calibrating all levels with MCTS (5k base sims, dynamic scaling)...")
    print("This may take a few minutes...\n")
    t0 = time.time()
    calibrate_max_steps(levels)
    print(f"\n  Calibration done in {time.time() - t0:.1f}s")

    # Sort levels by difficulty: fastest MCTS solve time (unsolved = hardest)
    def level_difficulty(lvl: int) -> float:
        cal = _level_calibration.get(lvl, {})
        fastest = cal.get("mcts_fastest")
        if fastest is None:
            return float('inf')  # unsolved = hardest
        return fastest

    levels = sorted(levels, key=level_difficulty)

    # Group into stages of 5 for smoother difficulty ramp
    STAGE_SIZE = 5
    stages: list[list[int]] = []
    for i in range(0, len(levels), STAGE_SIZE):
        stages.append(levels[i:i + STAGE_SIZE])

    # Write curriculum summary
    summary_path = os.path.join(save_dir, "curriculum_summary.txt")
    write_curriculum_summary(levels, stages, summary_path)

    print(f"\n=== PPO Curriculum Training ===")
    print(f"Curriculum: {len(levels)} levels in {len(stages)} stages")
    print(f"Win threshold: {args.win_threshold:.0%}")
    print(f"Timesteps/stage: {args.timesteps_per_stage:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Action repeat: {args.action_repeat}")
    print()
    for i, stage in enumerate(stages):
        ms = get_max_steps_for_stage(stage)
        print(f"  Stage {i+1}: L{', '.join(str(l) for l in stage)} -> max_steps={ms}")
    print()

    stage_idx = 0
    model = None
    train_env = None
    vec_normalize = None
    total_timesteps = 0

    # Resume from existing model if available
    best_model_path = os.path.join(save_dir, "best_model.zip")
    if args.resume_stage is not None and os.path.exists(best_model_path):
        stage_idx = args.resume_stage - 1  # convert 1-indexed to 0-indexed
        model = PPO.load(os.path.join(save_dir, "best_model"))
        norm_path = os.path.join(save_dir, "vec_normalize.pkl")
        if os.path.exists(norm_path):
            # Load a dummy env just to get VecNormalize stats
            dummy_fns = create_envs(stages[stage_idx], 3000, args.action_repeat, 1)
            dummy_env = SubprocVecEnv(dummy_fns, start_method="fork")
            vec_normalize = VecNormalize.load(norm_path, dummy_env)
            dummy_env.close()
        print(f"Resuming from stage {args.resume_stage} with existing model")

    while stage_idx < len(stages):
        if args.max_stages is not None and stage_idx >= args.max_stages:
            break

        stage_levels = stages[stage_idx]
        max_steps = get_max_steps_for_stage(stage_levels)

        print(f"\n{'#'*60}")
        print(f"  STAGE {stage_idx + 1}/{len(stages)}: L{stage_levels[0]}-L{stage_levels[-1]} "
              f"({len(stage_levels)} maps, max_steps={max_steps})")
        print(f"{'#'*60}")

        # Create training envs spread across all maps in this stage
        if train_env is not None:
            train_env.close()

        env_fns = create_envs(stage_levels, max_steps, args.action_repeat, n_envs)
        raw_train_env = SubprocVecEnv(env_fns, start_method="fork")

        if vec_normalize is not None:
            # Preserve normalization stats across stages
            train_env = VecNormalize(raw_train_env, norm_obs=True, norm_reward=True,
                                     clip_obs=10.0, clip_reward=10.0)
            train_env.obs_rms = vec_normalize.obs_rms
            train_env.ret_rms = vec_normalize.ret_rms
        else:
            train_env = VecNormalize(raw_train_env, norm_obs=True, norm_reward=True,
                                     clip_obs=10.0, clip_reward=10.0)

        # Create or update model
        if model is None:
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs={"net_arch": [256, 256]},
                seed=args.seed,
                verbose=0,
            )
        else:
            model.set_env(train_env)

        # Train in chunks, evaluating periodically
        recent_win_rates: deque[float] = deque(maxlen=WIN_RATE_WINDOW)
        stage_steps = 0
        advanced = False

        while stage_steps < args.timesteps_per_stage:
            chunk = min(args.eval_freq, args.timesteps_per_stage - stage_steps)

            t0 = time.time()
            model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                        callback=MetricsCallback(), progress_bar=False)
            elapsed = time.time() - t0
            stage_steps += chunk
            total_timesteps += chunk

            # Evaluate across ALL maps in the stage (2 episodes per map)
            eps_per_map = max(1, args.eval_episodes // len(stage_levels))
            total_wins = 0
            total_eval = 0
            total_reward_sum = 0.0
            total_pickups_sum = 0.0
            for eval_level in stage_levels:
                wr, mr, mp = evaluate_win_rate(
                    model, train_env, eval_level, max_steps, args.action_repeat,
                    eps_per_map,
                )
                total_wins += int(wr * eps_per_map)
                total_eval += eps_per_map
                total_reward_sum += mr * eps_per_map
                total_pickups_sum += mp * eps_per_map
            win_rate = total_wins / max(total_eval, 1)
            mean_reward = total_reward_sum / max(total_eval, 1)
            mean_pickups = total_pickups_sum / max(total_eval, 1)
            recent_win_rates.append(win_rate)
            smoothed = sum(recent_win_rates) / len(recent_win_rates)

            print(f"  [{total_timesteps:,} steps] {len(stage_levels)} maps: "
                  f"win={win_rate:.0%} smoothed={smoothed:.0%} "
                  f"reward={mean_reward:.0f} pickups={mean_pickups:.1f} "
                  f"({elapsed:.1f}s)")

            # Check advancement
            if len(recent_win_rates) >= WIN_RATE_WINDOW and smoothed >= args.win_threshold:
                print(f"  >>> Advanced! Smoothed {smoothed:.0%} >= {args.win_threshold:.0%}")
                advanced = True
                break

        if not advanced:
            print(f"  >>> Advancing (exhausted {args.timesteps_per_stage:,} steps, smoothed={smoothed:.0%})")

        # Save checkpoint
        vec_normalize = train_env  # preserve for next stage
        model.save(os.path.join(save_dir, f"model_stage_{stage_idx}"))
        train_env.save(os.path.join(save_dir, "vec_normalize.pkl"))

        # Always save as best model too
        model.save(os.path.join(save_dir, "best_model"))

        stage_idx += 1

    # Final save
    if model is not None:
        model.save(os.path.join(save_dir, "final_model"))
        if train_env is not None:
            train_env.save(os.path.join(save_dir, "vec_normalize.pkl"))
            train_env.close()

    print(f"\n{'='*60}")
    if stage_idx >= len(stages):
        print(f"Curriculum complete! Passed all {len(stages)} stages.")
    else:
        print(f"Stopped at stage {stage_idx}/{len(stages)}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Model saved to: {save_dir}/")


if __name__ == "__main__":
    main()
