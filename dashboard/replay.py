"""Run an agent on a level and capture frame-by-frame replay data."""

from __future__ import annotations

import math
import os

import numpy as np


def capture_replay(
    agent_type: str,
    level: int,
    max_steps: int = 3000,
    action_repeat: int = 5,
    model_path: str | None = None,
    num_simulations: int | None = None,
) -> dict:
    """Run one episode and return the full replay as a dict.

    For PPO agents, records every physics frame (not just every action_repeat
    group) so playback at 60fps looks smooth and natural.
    """
    if agent_type == "ppo":
        return _capture_ppo_replay(level, max_steps, action_repeat, model_path)

    return _capture_generic_replay(
        agent_type, level, max_steps, action_repeat, model_path, num_simulations
    )


def _capture_ppo_replay(
    level: int, max_steps: int, action_repeat: int, model_path: str | None
) -> dict:
    """PPO-specific replay: step the raw env every physics frame for smooth playback."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from spaceace.core.gym_wrapper import SpaceAceGymWrapper
    from spaceace.training.envs import StrategyWrapper, _build_strategies

    if model_path is None:
        for candidate in [
            "models/ppo/curriculum/best_model",
            f"models/{level}/best_model",
        ]:
            if os.path.exists(candidate + ".zip"):
                model_path = candidate
                break
        if model_path is None:
            raise FileNotFoundError("No PPO model found")

    # Build env with action_repeat=1 so we record every physics frame
    base_env = SpaceAceGymWrapper(level=level, max_steps=max_steps)
    obs_strategy, reward_strategy, pf = _build_strategies(
        level, max_steps, "path_augmented", "dense_shaped"
    )
    wrapped_env = StrategyWrapper(base_env, obs_strategy, reward_strategy, action_repeat=1, pathfinder=pf)
    vec_env = DummyVecEnv([lambda: wrapped_env])

    norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)

    model = PPO.load(model_path)

    raw_env = base_env.env
    geom = raw_env.get_map_geometry()
    walls = [[float(v) for v in seg] for seg in geom["map_lines"]]
    bounds = {k: float(v) for k, v in geom["bounds"].items()}
    pickups_initial = [[float(p[0]), float(p[1])] for p in geom["pickup_positions"]]

    frames = []
    pickup_events = []
    prev_remaining = len(pickups_initial)

    obs = vec_env.reset()
    step = 0
    current_action = None
    frames_since_decision = action_repeat  # force decision on first frame

    while True:
        # Make a new decision every action_repeat frames
        if frames_since_decision >= action_repeat:
            action, _ = model.predict(obs, deterministic=True)
            current_action = action[0]
            frames_since_decision = 0

        obs, reward, dones, infos = vec_env.step(action)
        step += 1
        frames_since_decision += 1

        raw_obs = raw_env.get_observation()
        remaining = int(raw_obs[16])

        pickup_states = list(raw_env.get_pickup_states())
        frames.append({
            "x": round(float(raw_obs[0]), 1),
            "y": round(float(raw_obs[1]), 1),
            "vx": round(float(raw_obs[2]), 1),
            "vy": round(float(raw_obs[3]), 1),
            "rotation": round(float(raw_obs[4]), 3),
            "action": [int(current_action[0]), int(current_action[1]), int(current_action[2])],
            "pickups_remaining": remaining,
            "pickup_collected": pickup_states,
            "reward": round(float(reward[0]), 3),
        })

        if remaining < prev_remaining:
            pickup_events.append({"step": step, "remaining": remaining})
        prev_remaining = remaining

        if bool(dones[0]):
            break

    info = infos[0]
    if info.get("level_completed"):
        outcome = "completed"
    elif info.get("ship_exploded"):
        outcome = "crashed"
    else:
        outcome = "truncated"

    vec_env.close()

    return {
        "agent": "ppo",
        "level": level,
        "outcome": outcome,
        "total_steps": step,
        "walls": walls,
        "bounds": bounds,
        "pickups_initial": pickups_initial,
        "frames": frames,
        "pickup_events": pickup_events,
    }


def _capture_generic_replay(
    agent_type: str,
    level: int,
    max_steps: int,
    action_repeat: int,
    model_path: str | None,
    num_simulations: int | None,
) -> dict:
    """Generic replay capture for non-PPO agents."""
    import spaceace.agents  # noqa: F401
    from spaceace.agents.base import AGENT_REGISTRY

    agent_cls = AGENT_REGISTRY[agent_type]
    agent = agent_cls()

    setup_kwargs = {}
    if model_path:
        setup_kwargs["model_path"] = model_path
    if num_simulations and agent_type in ("mcts", "alphazero"):
        setup_kwargs["num_simulations"] = num_simulations
    if agent_type in ("mcts", "alphazero"):
        setup_kwargs["action_repeat"] = action_repeat

    agent.setup(level=level, max_steps=max_steps, **setup_kwargs)
    agent.reset()

    raw_env = agent.get_raw_env()
    geom = raw_env.get_map_geometry()

    walls = [[float(v) for v in seg] for seg in geom["map_lines"]]
    bounds = {k: float(v) for k, v in geom["bounds"].items()}
    pickups_initial = [[float(p[0]), float(p[1])] for p in geom["pickup_positions"]]

    frames = []
    pickup_events = []
    prev_remaining = len(pickups_initial)

    step = 0
    while True:
        action, reward, terminated, truncated, info = agent.step()
        step += 1

        obs = raw_env.get_observation()
        remaining = int(obs[16])

        frames.append({
            "x": round(float(obs[0]), 1),
            "y": round(float(obs[1]), 1),
            "vx": round(float(obs[2]), 1),
            "vy": round(float(obs[3]), 1),
            "rotation": round(float(obs[4]), 3),
            "action": [int(action[0]), int(action[1]), int(action[2])],
            "pickups_remaining": remaining,
            "reward": round(float(reward), 3),
        })

        if remaining < prev_remaining:
            pickup_events.append({"step": step, "remaining": remaining})
        prev_remaining = remaining

        if terminated or truncated:
            break

    if info.get("level_completed"):
        outcome = "completed"
    elif info.get("ship_exploded"):
        outcome = "crashed"
    else:
        outcome = "truncated"

    return {
        "agent": agent_type,
        "level": level,
        "outcome": outcome,
        "total_steps": step,
        "walls": walls,
        "bounds": bounds,
        "pickups_initial": pickups_initial,
        "frames": frames,
        "pickup_events": pickup_events,
    }
