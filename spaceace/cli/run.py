"""`run.py` command — dispatch any registered agent via AGENT_REGISTRY."""

from __future__ import annotations

import argparse
import json
import time

import spaceace.agents  # noqa: F401 — eager-imports built-in agents
from spaceace.agents.base import AGENT_REGISTRY
from spaceace.agents import load_agent_module
from spaceace.core.viz import VisualRenderer, extract_game_info


def _save_ghost_if_best(level: int, ghost_type: str, step_count: int, frames: list[dict]) -> None:
    """If this run's completion time beats the stored ghost for (level, ghost_type),
    overwrite it in the dashboard DB. Mirrors scripts/capture_ai_ghost.py's save
    path — same frame format, same down-sample cadence (~10fps), same "faster wins"
    rule — so run.py and the capture script populate a consistent ghost table."""
    try:
        from dashboard.db import get_db, init_db
    except Exception as e:
        print(f"  [ghost] dashboard.db unavailable ({e}); skipping save")
        return

    time_seconds = step_count / 60.0
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
    try:
        existing = db.execute(
            "SELECT time_seconds FROM ghost_replays WHERE level = ? AND ghost_type = ?",
            (level, ghost_type),
        ).fetchone()
        if existing and existing["time_seconds"] <= time_seconds:
            print(f"  [ghost] existing {ghost_type} ghost for level {level} is faster "
                  f"({existing['time_seconds']:.2f}s ≤ {time_seconds:.2f}s), not saving")
            return
        db.execute(
            """INSERT OR REPLACE INTO ghost_replays
               (level, ghost_type, steps, time_seconds, frames_json)
               VALUES (?, ?, ?, ?, ?)""",
            (level, ghost_type, len(ghost_frames), time_seconds,
             json.dumps(ghost_frames)),
        )
        db.commit()
        prev = f" (prev {existing['time_seconds']:.2f}s)" if existing else ""
        print(f"  [ghost] saved {ghost_type} for level {level}: "
              f"{len(ghost_frames)} frames, {time_seconds:.2f}s{prev}")
    finally:
        db.close()


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
    p.add_argument("--beam-width", type=int, default=1000,
                   help="Beam width for beam search agent (default 1000)")
    p.add_argument("--step-penalty", type=float, default=0.01,
                   help="Step penalty for beam search scoring (default 0.01)")
    p.add_argument("--no-optimize", action="store_true",
                   help="Skip trajectory optimization phase for beam search")
    p.add_argument("--action-repeat", type=int, default=3,
                   help="Action repeat frames for beam/MCTS (default 3)")
    p.add_argument("--ar-depth-bonus", type=int, default=0,
                   help="MCTS: additional action_repeat frames per tree depth level. "
                   "0 (default) = constant, 1-2 = longer horizon deep in tree.")
    p.add_argument("--ar-max", type=int, default=20,
                   help="MCTS: cap on depth-scaled action_repeat (default 20)")
    p.add_argument("--thrust-bias", type=float, default=0.0,
                   help="MCTS: additive UCT bonus for thrust-on actions (1,3,5). "
                   "Biases the tree toward keeping thrust on. Typical: 0.2-0.6.")
    p.add_argument("--thrust-bias-safe-dist", type=float, default=0.0,
                   help="MCTS: nearest-wall distance (px) at which --thrust-bias reaches "
                   "full strength. Below this, bias fades linearly to 0. 0 (default) "
                   "disables scaling — constant bias everywhere, including tight corners. "
                   "Set >0 only if the agent is thrusting into walls.")
    p.add_argument("--rollout-frames", type=int, default=0,
                   help="MCTS leaf policy rollout length (frames). At each expanded leaf, "
                   "runs N frames of argmax-prior rollout before heuristic eval. "
                   "Typical: 20-40. 0 (default) disables.")
    p.add_argument("--ee-check-every", type=int, default=0,
                   help="MCTS adaptive early-exit: check root visit distribution every N "
                   "sims; stop early when one action dominates. 0 (default) disables. "
                   "Typical: 500-1000.")
    p.add_argument("--ee-visit-frac", type=float, default=0.6,
                   help="Minimum root visit fraction of the best action to trigger "
                   "early-exit (default 0.6).")
    p.add_argument("--ee-q-gap", type=float, default=0.0,
                   help="Minimum mean-value gap between best and runner-up to trigger "
                   "early-exit (default 0.0 = visit-fraction only).")
    p.add_argument("--widen-k", type=float, default=0.0,
                   help="MCTS progressive widening coefficient. 0 (default) = disabled. "
                   "Typical: 1.0-1.5. Shrinks effective branching factor at shallow-visit "
                   "nodes so promising lines grow deeper for the same sim budget.")
    # A* planner options
    p.add_argument("--astar-action-repeat", type=int, default=4,
                   help="Frames per macro-action for A* planner (default 4)")
    p.add_argument("--astar-pos-bucket", type=float, default=8.0,
                   help="Position bucket size (px) for A* state canonicalization (default 8)")
    p.add_argument("--astar-vel-bucket", type=float, default=8.0,
                   help="Velocity bucket size (px/s) for A* (default 8)")
    p.add_argument("--astar-rot-bucket-deg", type=float, default=10.0,
                   help="Rotation bucket size (degrees) for A* (default 10)")
    p.add_argument("--astar-leg-max-expansions", type=int, default=200_000,
                   help="Per-leg expansion cap for A* inner solver (default 200000)")
    p.add_argument("--astar-leg-time-limit", type=float, default=60.0,
                   help="Per-leg wall-clock time limit for A* inner solver, seconds (default 60)")
    p.add_argument("--astar-heuristic-weight", type=float, default=1.0,
                   help="A* heuristic multiplier. 1.0=admissible, >1.0=weighted (faster, bounded-suboptimal)")
    # mcts_rewind agent options
    p.add_argument("--rewind-budget", type=int, default=8,
                   help="mcts_rewind: max rewinds per episode (default 8)")
    p.add_argument("--rewind-history", type=int, default=40,
                   help="mcts_rewind: how many prior checkpoints to keep (default 40)")
    p.add_argument("--rewind-stuck", type=int, default=180,
                   help="mcts_rewind: frames without pickup progress before rewind fires (default 180)")
    p.add_argument("--rewind-regret", type=float, default=0.35,
                   help="mcts_rewind: value drop vs baseline that triggers rewind (default 0.35)")
    p.add_argument("--rewind-num-simulations", type=int, default=0,
                   help="mcts_rewind: sims on rewind search. 0 (default) = use --num-simulations")
    p.add_argument("--save-ghost", dest="save_ghost", action="store_true", default=True,
                   help="On completed episodes, save the run as a ghost in the dashboard "
                   "DB if it's faster than the existing ghost for this level. "
                   "On by default; disable with --no-save-ghost.")
    p.add_argument("--no-save-ghost", dest="save_ghost", action="store_false",
                   help="Disable ghost saving.")
    p.add_argument("--ghost-label", type=str, default=None,
                   help="ghost_type label for saved ghosts (default: the agent name, "
                   "e.g. 'mcts'). Same convention as scripts/capture_ai_ghost.py --label.")
    p.add_argument("--screenshots", type=int, nargs="?", const=30, default=0,
                   metavar="N", help="Save screenshots every N frames (default 30 if flag given). "
                   "Also captures on crash/completion. Press S for manual capture.")
    p.add_argument("--screenshot-dir", type=str, default="screenshots",
                   help="Directory for screenshots (default: screenshots/)")
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
        "beam_width": args.beam_width,
        "step_penalty": args.step_penalty,
        "action_repeat": args.action_repeat,
        "action_repeat_depth_bonus": args.ar_depth_bonus,
        "action_repeat_max": args.ar_max,
        "widen_k": args.widen_k,
        "rollout_frames": args.rollout_frames,
        "early_exit_check_every": args.ee_check_every,
        "early_exit_visit_frac": args.ee_visit_frac,
        "early_exit_q_gap": args.ee_q_gap,
        "thrust_bias": args.thrust_bias,
        "thrust_bias_safe_dist": args.thrust_bias_safe_dist,
        "optimize": not args.no_optimize,
        "astar_action_repeat": args.astar_action_repeat,
        "astar_pos_bucket": args.astar_pos_bucket,
        "astar_vel_bucket": args.astar_vel_bucket,
        "astar_rot_bucket_deg": args.astar_rot_bucket_deg,
        "astar_leg_max_expansions": args.astar_leg_max_expansions,
        "astar_leg_time_limit_s": args.astar_leg_time_limit,
        "astar_heuristic_weight": args.astar_heuristic_weight,
        "rewind_budget": args.rewind_budget,
        "rewind_history": args.rewind_history,
        "rewind_stuck": args.rewind_stuck,
        "rewind_regret": args.rewind_regret,
        "rewind_num_simulations": (
            args.rewind_num_simulations if args.rewind_num_simulations > 0 else args.num_simulations
        ),
    }
    if args.model:
        kwargs["model_path"] = args.model

    agent.setup(level=args.level, max_steps=args.max_steps, **kwargs)

    renderer = None if args.headless else VisualRenderer(
        width=1200, height=800, screenshot_dir=args.screenshot_dir,
    )
    if renderer and args.screenshots:
        renderer.enable_auto_screenshots(args.screenshots)
        print(f"Screenshots enabled: every {args.screenshots} frames → {args.screenshot_dir}/")
    print(f"Running {args.agent} agent on level {args.level} for {args.episodes} episodes")

    try:
        ghost_label = args.ghost_label or args.agent
        for episode in range(args.episodes):
            agent.reset()
            if renderer:
                renderer.set_episode(episode)
            total_reward = 0.0
            step_count = 0
            frames: list[dict] = []
            raw_env = agent.get_raw_env()
            while True:
                action, reward, terminated, truncated, info = agent.step()
                total_reward += reward
                step_count += 1

                if args.save_ghost:
                    obs = raw_env.get_observation()
                    frames.append({
                        "x": round(float(obs[0]), 1),
                        "y": round(float(obs[1]), 1),
                        "rotation": round(float(obs[4]), 3),
                        "thrusting": int(action[2]) > 0,
                    })

                if renderer:
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
                    if args.save_ghost and info.get("level_completed"):
                        _save_ghost_if_best(args.level, ghost_label, step_count, frames)
                    if renderer and args.screenshots:
                        renderer.save_screenshot(status.lower())
                    if renderer:
                        time.sleep(2)
                    break
    finally:
        agent.close()
        if renderer:
            renderer.close()


if __name__ == "__main__":
    main()
