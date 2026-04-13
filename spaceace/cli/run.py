"""`run.py` command — dispatch any registered agent via AGENT_REGISTRY."""

from __future__ import annotations

import argparse
import time

import spaceace.agents  # noqa: F401 — eager-imports built-in agents
from spaceace.agents.base import AGENT_REGISTRY
from spaceace.agents import load_agent_module
from spaceace.core.viz import VisualRenderer, extract_game_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a SpaceAce agent")
    p.add_argument("--agent", type=str, default="random",
                   help=f"Agent type. Built-ins: {', '.join(sorted(AGENT_REGISTRY))}")
    p.add_argument("--agent-module", type=str, default=None,
                   help="Dotted path to a module that registers a custom agent")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--num-simulations", type=int, default=200)
    p.add_argument("--exploration", type=float, default=1.41)
    p.add_argument("--momentum-pathfinder", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.agent_module:
        load_agent_module(args.agent_module)

    if args.agent not in AGENT_REGISTRY:
        raise SystemExit(
            f"Unknown agent '{args.agent}'. Available: {', '.join(sorted(AGENT_REGISTRY))}"
        )
    agent = AGENT_REGISTRY[args.agent]()

    kwargs = {
        "num_simulations": args.num_simulations,
        "exploration_constant": args.exploration,
        "momentum_pathfinder": args.momentum_pathfinder,
    }
    if args.model:
        kwargs["model_path"] = args.model

    agent.setup(level=args.level, max_steps=args.max_steps, **kwargs)

    renderer = None if args.headless else VisualRenderer(width=1200, height=800)
    print(f"Running {args.agent} agent on level {args.level} for {args.episodes} episodes")

    try:
        for episode in range(args.episodes):
            agent.reset()
            total_reward = 0.0
            step_count = 0
            while True:
                action, reward, terminated, truncated, info = agent.step()
                total_reward += reward
                step_count += 1

                if renderer:
                    raw_env = agent.get_raw_env()
                    raw_obs = raw_env.get_observation()
                    game_state, pickups, map_bounds = extract_game_info(raw_obs, info, raw_env)
                    game_state["level"] = args.level
                    debug_path = None
                    mcts_debug = None
                    if hasattr(agent, "_mcts"):
                        state = raw_env.save_state()
                        debug_path = agent._mcts.get_debug_path(state)
                    if hasattr(agent, "debug_info") and agent.debug_info:
                        mcts_debug = agent.debug_info
                    if not renderer.render_frame(
                        game_state, info, total_reward, action, pickups, map_bounds, debug_path, mcts_debug
                    ):
                        return
                    renderer.wait_for_fps(args.fps)

                if terminated or truncated:
                    status = (
                        "COMPLETED" if info.get("level_completed")
                        else "CRASHED" if info.get("ship_exploded")
                        else "TRUNCATED"
                    )
                    print(f"  Episode {episode+1}: {status} | steps={step_count} reward={total_reward:.1f}")
                    if renderer:
                        time.sleep(2)
                    break
    finally:
        agent.close()
        if renderer:
            renderer.close()


if __name__ == "__main__":
    main()
