"""Train the low-level waypoint pilot for the HRL agent using PPO.

PPO is on-policy and benefits from many parallel environments. Uses
Discrete(8) action space via ActionFlattenWrapper with PBRS reward.

Usage:
    uv run python -m spaceace.agents.hrl.train_pilot --levels 4000-4099 --timesteps 500000
    uv run python -m spaceace.agents.hrl.train_pilot --levels 4000 4010 4020 --timesteps 100000
"""

import argparse
import multiprocessing
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from spaceace.core.gym_wrapper import SpaceAceGymWrapper
from spaceace.agents.hrl.waypoint_env import (
    WaypointPilotEnv,
    WaypointMetricsCallback,
    make_waypoint_env,
)


def parse_level_spec(spec: str) -> list:
    """Parse level spec like '4000-4099' or '4000' into list of ints."""
    if '-' in spec:
        start, end = spec.split('-')
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
    args = parse_args()

    # Parse level specs
    levels = []
    for spec in args.levels:
        levels.extend(parse_level_spec(spec))
    levels = sorted(set(levels))

    save_dir = args.model_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    # PPO is on-policy — benefits from many parallel envs
    n_envs = min(16, max(8, len(levels)))

    print(f"=== HRL Waypoint Pilot Training (PPO) ===")
    print(f"Levels: {levels[0]}-{levels[-1]} ({len(levels)} total)")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Max steps/waypoint: {args.max_steps}")
    print(f"Action repeat: {args.action_repeat}")
    print(f"Envs: {n_envs}")
    print(f"Save dir: {save_dir}")
    print()

    # --- Training environment ---
    def make_train_envs():
        envs = []
        for i in range(n_envs):
            level = levels[i % len(levels)]
            envs.append(make_waypoint_env(level, args.max_steps, args.action_repeat,
                                          flatten_actions=True))
        return envs

    train_env = DummyVecEnv(make_train_envs())
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False,
                             clip_obs=10.0)

    # --- Eval environment ---
    eval_level = levels[len(levels) // 2]  # middle difficulty
    eval_env = DummyVecEnv([make_waypoint_env(eval_level, args.max_steps, args.action_repeat,
                                               flatten_actions=True)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # --- Model ---
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=train_env)
    else:
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
            policy_kwargs={"net_arch": [256, 256]},
            tensorboard_log=args.tensorboard_dir,
            seed=args.seed,
            verbose=1,
        )

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )
    metrics_callback = WaypointMetricsCallback()

    # --- Train ---
    steps_k = args.timesteps // 1000
    run_name = f"hrl_ppo_corridors_{steps_k}k"
    print(f"Starting training (run: {run_name})...")
    start = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, metrics_callback],
        tb_log_name=run_name,
        progress_bar=True,
    )
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # --- Save final model + normalization stats ---
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    train_env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    print(f"Saved final model to {final_path}.zip")
    print(f"Saved normalization stats to {save_dir}/vec_normalize.pkl")

    # --- Final evaluation ---
    eval_levels = [levels[0], levels[len(levels)//4], levels[len(levels)//2],
                   levels[3*len(levels)//4], levels[-1]]
    print(f"\n=== Final Evaluation (10 eps each on {eval_levels}) ===")

    for lvl in eval_levels:
        raw_base = SpaceAceGymWrapper(level=lvl, max_steps=args.max_steps)
        from spaceace.agents.hrl.waypoint_env import ActionFlattenWrapper
        raw_wp = WaypointPilotEnv(raw_base, level=lvl, max_steps=args.max_steps,
                                   action_repeat=args.action_repeat)
        raw_env = ActionFlattenWrapper(raw_wp)
        raw_vec = DummyVecEnv([lambda: Monitor(raw_env)])
        raw_vec = VecNormalize.load(os.path.join(save_dir, "vec_normalize.pkl"), raw_vec)
        raw_vec.training = False
        raw_vec.norm_reward = False

        reached = crashed = 0
        total_steps = 0
        for _ in range(10):
            obs = raw_vec.reset()
            done = False
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, r, d, i = raw_vec.step(a)
                done = d[0]
            if raw_wp.last_episode_waypoint_reached: reached += 1
            if raw_wp.last_episode_crashed: crashed += 1
            total_steps += raw_wp.last_episode_steps

        print(f"  Level {lvl}: WP={reached}/10 CR={crashed}/10 steps={total_steps//10}")
        raw_vec.close()

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
