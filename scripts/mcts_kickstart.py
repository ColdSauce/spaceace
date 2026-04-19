#!/usr/bin/env python3
"""Manually run MCTS kickstart on the current training model.

Usage:
    uv run python scripts/mcts_kickstart.py --level 3029
    uv run python scripts/mcts_kickstart.py --level 3029 --episodes 10 --sims 5000
"""

import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO

import spaceace_rl
from spaceace.core.gym_wrapper import SpaceAceGymWrapper
from spaceace.strategies.actions import ALL_ACTIONS, ACTION_NAMES
from spaceace.training.envs import StrategyWrapper, _build_strategies


def collect_demos(level, max_steps, action_repeat, num_episodes, num_sims):
    demos = []
    for ep in range(num_episodes):
        base_env = SpaceAceGymWrapper(level=level, max_steps=max_steps)
        obs_strategy, reward_strategy, pf = _build_strategies(
            level, max_steps, "path_augmented", "dense_shaped"
        )
        wrapped = StrategyWrapper(
            base_env, obs_strategy, reward_strategy,
            action_repeat=action_repeat, pathfinder=pf,
        )
        mcts = spaceace_rl.PyMCTSEngine(level, max_steps, False)
        raw_env = base_env.env

        pa_obs = wrapped.reset()[0]

        done = False
        ep_steps = 0
        while not done:
            state = raw_env.save_state()
            action_idx = mcts.search(state, num_sims, action_repeat, 1.41, 0.99)
            action = ALL_ACTIONS[action_idx]

            demos.append((pa_obs.copy(), action.copy()))

            pa_obs, reward, terminated, truncated, info = wrapped.step(action)
            done = terminated or truncated
            ep_steps += 1
            if ep_steps > max_steps // action_repeat:
                break

        metrics = info.get("episode_metrics", {})
        completed = metrics.get("completed", False)
        pickups = metrics.get("pickups_collected", 0)
        print(f"  MCTS ep {ep+1}/{num_episodes}: "
              f"{'COMPLETED' if completed else 'FAILED'} | "
              f"{ep_steps} steps | {pickups} pickups")

    return demos


def behavioral_cloning(model, demos, epochs=3, lr=1e-3):
    obs_list, actions_list = zip(*demos)
    policy = model.policy
    device = policy.device

    obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(np.array(actions_list), dtype=torch.long, device=device)

    bc_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        indices = torch.randperm(len(obs_tensor), device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices), 256):
            batch_idx = indices[start:start + 256]
            batch_obs = obs_tensor[batch_idx]
            batch_actions = actions_tensor[batch_idx]

            _, log_prob, entropy = policy.evaluate_actions(batch_obs, batch_actions)
            bc_loss = -log_prob.mean() - 0.01 * entropy.mean()

            bc_optimizer.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            bc_optimizer.step()

            total_loss += bc_loss.item()
            n_batches += 1

        print(f"  BC epoch {epoch+1}/{epochs}: loss={total_loss / max(n_batches, 1):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--model", type=str, default="models/curriculum/latest_model")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--sims", type=int, default=2000)
    p.add_argument("--bc-epochs", type=int, default=3)
    p.add_argument("--bc-lr", type=float, default=1e-3)
    args = p.parse_args()

    model_path = args.model
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    print(f"\nCollecting MCTS demos on level {args.level} ({args.episodes} episodes, {args.sims} sims)...")
    demos = collect_demos(
        args.level, args.max_steps, args.action_repeat, args.episodes, args.sims
    )

    if not demos:
        print("No demos collected!")
        return

    print(f"\nRunning behavioral cloning ({len(demos)} steps)...")
    behavioral_cloning(model, demos, epochs=args.bc_epochs, lr=args.bc_lr)

    print(f"\nSaving model back to {model_path}")
    model.save(model_path)
    print("Done! The training run will pick up the updated model on next checkpoint load.")


if __name__ == "__main__":
    main()
