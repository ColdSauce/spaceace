"""Run an agent on a level and save the resulting ghost to the dashboard DB."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.db import get_db, init_db  # noqa: E402
from spaceace.agents.base import AGENT_REGISTRY  # noqa: E402
import spaceace.agents  # noqa: E402,F401
from spaceace.ghost_actions import action_to_index, write_sidecar_if_best  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, choices=sorted(AGENT_REGISTRY))
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--num-simulations", type=int, default=10000)
    p.add_argument("--exploration", type=float, default=1.41)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--beam-width", type=int, default=2000)
    p.add_argument("--step-penalty", type=float, default=0.01)
    p.add_argument("--no-optimize", action="store_true")
    p.add_argument("--label", default=None,
                   help="Optional label to distinguish ghosts (stored as ghost_type)")
    p.add_argument("--thrust-bias", type=float, default=0.0)
    p.add_argument("--thrust-bias-safe-dist", type=float, default=0.0)
    p.add_argument("--rewind-budget", type=int, default=8)
    p.add_argument("--rewind-history", type=int, default=40)
    p.add_argument("--rewind-stuck", type=int, default=180)
    p.add_argument("--rewind-regret", type=float, default=0.35)
    p.add_argument("--tas-path", type=str, default=None)
    p.add_argument("--tas-label", type=str, default="ai")
    p.add_argument("--tas-validate", action="store_true")
    args = p.parse_args()

    init_db()

    agent_cls = AGENT_REGISTRY[args.agent]
    agent = agent_cls()

    setup_kwargs = {}
    if args.agent == "mcts":
        setup_kwargs["num_simulations"] = args.num_simulations
        setup_kwargs["exploration_constant"] = args.exploration
        setup_kwargs["action_repeat"] = args.action_repeat
        setup_kwargs["thrust_bias"] = args.thrust_bias
        setup_kwargs["thrust_bias_safe_dist"] = args.thrust_bias_safe_dist
    elif args.agent == "mcts_rewind":
        setup_kwargs["num_simulations"] = args.num_simulations
        setup_kwargs["exploration_constant"] = args.exploration
        setup_kwargs["action_repeat"] = args.action_repeat
        setup_kwargs["thrust_bias"] = args.thrust_bias
        setup_kwargs["thrust_bias_safe_dist"] = args.thrust_bias_safe_dist
        setup_kwargs["rewind_budget"] = args.rewind_budget
        setup_kwargs["rewind_history"] = args.rewind_history
        setup_kwargs["rewind_stuck"] = args.rewind_stuck
        setup_kwargs["rewind_regret"] = args.rewind_regret
    elif args.agent == "beam":
        setup_kwargs["beam_width"] = args.beam_width
        setup_kwargs["step_penalty"] = args.step_penalty
        setup_kwargs["action_repeat"] = args.action_repeat
        setup_kwargs["optimize"] = not args.no_optimize
    elif args.agent == "tas":
        setup_kwargs["tas_path"] = args.tas_path
        setup_kwargs["tas_label"] = args.tas_label
        setup_kwargs["tas_validate"] = args.tas_validate

    print(f"[{args.agent}] level={args.level} kwargs={setup_kwargs}", flush=True)
    t0 = time.time()
    agent.setup(level=args.level, max_steps=args.max_steps, **setup_kwargs)
    agent.reset()
    setup_elapsed = time.time() - t0
    print(f"[{args.agent}] setup done in {setup_elapsed:.1f}s", flush=True)

    raw_env = agent.get_raw_env()
    frames = []
    action_indices = []
    last_action_tick = 0
    step = 0
    info = {}

    t0 = time.time()
    while True:
        action, reward, terminated, truncated, info = agent.step()
        step += 1
        obs = raw_env.get_observation()
        # Record the true physics-tick count. Agents like mcts_rewind advance
        # multiple ticks per agent.step(), so the decision index is not a
        # reliable time base.
        tick = int(info.get("step_count", step))
        delta_ticks = max(0, tick - last_action_tick)
        if delta_ticks:
            action_indices.extend([action_to_index(action)] * delta_ticks)
        last_action_tick = tick
        frames.append({
            "x": round(float(obs[0]), 1),
            "y": round(float(obs[1]), 1),
            "rotation": round(float(obs[4]), 3),
            "thrusting": int(action[2]) > 0,
            "tick": tick,
        })
        if terminated or truncated:
            break
    play_elapsed = time.time() - t0

    if info.get("level_completed"):
        outcome = "completed"
    elif info.get("ship_exploded"):
        outcome = "crashed"
    else:
        outcome = "truncated"

    tick_count = int(info.get("step_count", step))
    time_seconds = tick_count / 60.0
    print(f"[{args.agent}] outcome={outcome} decisions={step} ticks={tick_count} "
          f"game_time={time_seconds:.2f}s play_elapsed={play_elapsed:.1f}s", flush=True)

    if outcome != "completed":
        print(f"[{args.agent}] did not complete, not saving ghost", flush=True)
        return 1

    # Build compact ghost frames with time (downsample to ~10fps by physics ticks)
    ghost_frames = []
    target_stride = 6
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

    ghost_type = args.label if args.label else "ai"

    db = get_db()
    # Only save if faster than existing ghost of this type
    existing = db.execute(
        "SELECT time_seconds FROM ghost_replays WHERE level = ? AND ghost_type = ?",
        (args.level, ghost_type),
    ).fetchone()
    if existing and existing["time_seconds"] <= time_seconds:
        print(f"[{args.agent}] existing {ghost_type} ghost is faster "
              f"({existing['time_seconds']:.2f}s <= {time_seconds:.2f}s), not saving",
              flush=True)
        db.close()
        write_sidecar_if_best(args.level, ghost_type, action_indices, tick_count)
        return 0

    db.execute(
        """INSERT OR REPLACE INTO ghost_replays
           (level, ghost_type, steps, time_seconds, frames_json)
           VALUES (?, ?, ?, ?, ?)""",
        (args.level, ghost_type, len(ghost_frames), time_seconds,
         json.dumps(ghost_frames)),
    )
    db.commit()
    db.close()
    print(f"[{args.agent}] saved ghost (type={ghost_type}): "
          f"{len(ghost_frames)} frames, {time_seconds:.2f}s", flush=True)
    write_sidecar_if_best(args.level, ghost_type, action_indices, tick_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
