"""`run.py` command — dispatch any registered agent via AGENT_REGISTRY."""

from __future__ import annotations

import argparse
import json
import time

import spaceace.agents  # noqa: F401 — eager-imports built-in agents
from spaceace.agents.base import AGENT_REGISTRY
from spaceace.agents import load_agent_module
from spaceace.core.viz import VisualRenderer, extract_game_info
from spaceace.ghost_actions import action_to_index, write_sidecar_if_best


def _save_ghost_if_best(
    level: int,
    ghost_type: str,
    tick_count: int,
    frames: list[dict],
    action_indices: list[int] | None = None,
) -> None:
    """If this run's completion time beats the stored ghost for (level, ghost_type),
    overwrite it in the dashboard DB. Mirrors scripts/capture_ai_ghost.py's save
    path — same frame format, same down-sample cadence (~10fps), same "faster wins"
    rule — so run.py and the capture script populate a consistent ghost table.

    `tick_count` is the true physics-tick count (one tick = 1/60s). Each entry in
    `frames` must include a "tick" field recording the physics-tick count at
    which it was captured — not the agent-decision index — because agents like
    mcts_rewind advance multiple ticks per agent.step() call. Timestamping by
    decision index made those ghosts play back N× too fast."""
    try:
        from dashboard.db import get_db, init_db
    except Exception as e:
        print(f"  [ghost] dashboard.db unavailable ({e}); skipping save")
        return

    time_seconds = tick_count / 60.0
    ghost_frames = []
    target_stride = 6  # ~10 Hz playback
    next_emit_tick = 0
    last_idx = len(frames) - 1
    for i, f in enumerate(frames):
        tick = int(f.get("tick", i))
        if tick >= next_emit_tick or i == last_idx:
            ghost_frames.append({
                "x": f["x"], "y": f["y"],
                "rotation": f["rotation"],
                "thrusting": f["thrusting"],
                "time": round(tick / 60.0, 3),
            })
            next_emit_tick = tick + target_stride

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
        else:
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

    if action_indices is not None:
        write_sidecar_if_best(level, ghost_type, action_indices, tick_count)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a SpaceAce agent")
    p.add_argument("--agent", type=str, default="random",
                   help=f"Agent type. Built-ins: {', '.join(sorted(AGENT_REGISTRY))}")
    p.add_argument("--agent-module", type=str, default=None,
                   help="Dotted path to a module that registers a custom agent")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--headless", action="store_true")
    # TAS replay options
    p.add_argument("--tas-path", type=str, default=None,
                   help="Exact action trace JSON to replay with --agent tas. "
                   "Default: ghost_actions/L<level>_<tas-label>.json.")
    p.add_argument("--tas-label", type=str, default="tas",
                   help="Ghost action sidecar label for --agent tas (default: tas)")
    p.add_argument("--tas-validate", action="store_true",
                   help="Replay the TAS trace once during setup and fail if it does not complete.")
    # Ace agent options
    p.add_argument("--ace-width", type=int, default=40_000,
                   help="Beam width when --agent ace has to plan from scratch (default 40000)")
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
        "tas_path": args.tas_path,
        "tas_label": args.tas_label,
        "tas_validate": args.tas_validate,
        "ace_width": args.ace_width,
    }

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
            action_indices: list[int] = []
            last_action_tick = 0
            raw_env = agent.get_raw_env()
            while True:
                action, reward, terminated, truncated, info = agent.step()
                total_reward += reward
                step_count += 1

                if args.save_ghost:
                    obs = raw_env.get_observation()
                    tick = int(info.get("step_count", step_count))
                    delta_ticks = max(0, tick - last_action_tick)
                    if delta_ticks:
                        action_indices.extend([action_to_index(action)] * delta_ticks)
                    last_action_tick = tick
                    frames.append({
                        "x": round(float(obs[0]), 1),
                        "y": round(float(obs[1]), 1),
                        "rotation": round(float(obs[4]), 3),
                        "thrusting": int(action[2]) > 0,
                        # True physics-tick count — required so agents like
                        # mcts_rewind (which step multiple ticks per call)
                        # produce correctly-timed ghost replays.
                        "tick": tick,
                    })

                if renderer:
                    raw_obs = raw_env.get_observation()
                    game_state, pickups, map_bounds = extract_game_info(raw_obs, info, raw_env)
                    game_state["level"] = args.level
                    debug_info = getattr(agent, "debug_info", None) or None
                    if not renderer.render_frame(
                        game_state, info, total_reward, action, pickups, map_bounds, None, debug_info
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
                        tick_count = int(info.get("step_count", step_count))
                        _save_ghost_if_best(
                            args.level, ghost_label, tick_count, frames,
                            action_indices=action_indices,
                        )
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
