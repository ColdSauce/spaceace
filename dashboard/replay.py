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
    """Run one episode and return the full replay as a dict."""
    return _capture_generic_replay(
        agent_type, level, max_steps, action_repeat, model_path, num_simulations
    )


def _capture_generic_replay(
    agent_type: str,
    level: int,
    max_steps: int,
    action_repeat: int,
    model_path: str | None,
    num_simulations: int | None,
) -> dict:
    """Run a registered agent and record every frame."""
    import spaceace.agents  # noqa: F401
    from spaceace.agents.base import AGENT_REGISTRY

    agent_cls = AGENT_REGISTRY[agent_type]
    agent = agent_cls()

    setup_kwargs = {}

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
