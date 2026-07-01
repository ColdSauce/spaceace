"""Compare exact agent pickup segments against the human ghost.

The dashboard ghost table stores human runs as position frames, not exact
actions, so human pickup times are estimates from proximity to each pickup.
Agent timings are collected by replaying through the real engine and watching
the actual pickup-state bits change.

Examples:
    uv run python scripts/compare_agent_human_segments.py --level 7 --agent mcts
    uv run python scripts/compare_agent_human_segments.py --level 7 --agent mcts --num-simulations 1000 --action-repeat 5 --thrust-bias 1.0
    uv run python scripts/compare_agent_human_segments.py --level 7 --trace-json /tmp/spaceace_l7_tas_source.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import spaceace_rl  # noqa: E402

import spaceace.agents  # noqa: E402,F401
from spaceace.agents.base import AGENT_REGISTRY  # noqa: E402
from spaceace.core.env import SpaceAceDirectEnv  # noqa: E402
from spaceace.ghost_actions import action_to_index, dump_action_file, load_action_file  # noqa: E402
from spaceace.strategies.actions import ALL_ACTIONS  # noqa: E402

PICKUP_RADIUS = 46.5  # PICKUP_RADIUS + SHIP_COLLISION_RADIUS in src/real_physics.rs


@dataclass
class PickupEvent:
    pickup_idx: int
    tick: int
    time_s: float
    x: float
    y: float
    speed: float


@dataclass
class SegmentStats:
    t0: float
    t1: float
    duration: float
    mean_speed: float
    p90_speed: float
    max_speed: float
    thrust_fraction: float


def load_ghost(level: int, ghost_type: str) -> tuple[list[dict], float]:
    from dashboard.db import get_db, init_db

    init_db()
    db = get_db()
    try:
        row = db.execute(
            "SELECT frames_json, time_seconds FROM ghost_replays "
            "WHERE level = ? AND ghost_type = ?",
            (level, ghost_type),
        ).fetchone()
    finally:
        db.close()
    if row is None:
        raise ValueError(f"no {ghost_type} ghost for level {level}")
    frames = json.loads(row["frames_json"])
    return frames, float(row["time_seconds"])


def pickup_coords(level: int) -> list[tuple[float, float]]:
    pf = spaceace_rl.PyPathfinder(level, "grid")
    return [(float(x), float(y)) for x, y in pf.get_pickup_coords()]


def _speed_series(frames: list[dict], t0: float, t1: float) -> SegmentStats:
    speeds: list[float] = []
    thrust_count = 0
    frame_count = 0
    prev: Optional[dict] = None
    for frame in frames:
        t = float(frame["time"])
        if t < t0 or t > t1:
            continue
        frame_count += 1
        if frame.get("thrusting"):
            thrust_count += 1
        if prev is not None:
            dt = t - float(prev["time"])
            if dt > 0:
                dx = float(frame["x"]) - float(prev["x"])
                dy = float(frame["y"]) - float(prev["y"])
                speeds.append(math.hypot(dx, dy) / dt)
        prev = frame

    if speeds:
        ordered = sorted(speeds)
        p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
        mean_speed = sum(speeds) / len(speeds)
        max_speed = max(speeds)
    else:
        p90 = mean_speed = max_speed = 0.0

    return SegmentStats(
        t0=t0,
        t1=t1,
        duration=max(0.0, t1 - t0),
        mean_speed=mean_speed,
        p90_speed=p90,
        max_speed=max_speed,
        thrust_fraction=thrust_count / max(1, frame_count),
    )


def estimate_human_events(
    frames: list[dict],
    total_time: float,
    pickups: list[tuple[float, float]],
    radius: float,
) -> tuple[list[dict], list[int]]:
    """Return per-pickup timing estimates and collection order.

    For each pickup we report the first frame within `radius` when available,
    plus the nearest observed frame. If exactly one pickup lacks a radius hit,
    we assign it to the DB completion time with low confidence; browser ghosts
    often omit the final collision frame.
    """
    estimates: list[dict[str, Any]] = []
    for idx, (px, py) in enumerate(pickups):
        first_hit: Optional[tuple[float, float, float, float]] = None
        nearest: Optional[tuple[float, float, float, float]] = None
        for frame in frames:
            x = float(frame["x"])
            y = float(frame["y"])
            t = float(frame["time"])
            dist = math.hypot(x - px, y - py)
            if nearest is None or dist < nearest[0]:
                nearest = (dist, t, x, y)
            if first_hit is None and dist <= radius:
                first_hit = (dist, t, x, y)
        assert nearest is not None
        if first_hit is not None:
            dist, t, x, y = first_hit
            confidence = "within-radius"
        else:
            dist, t, x, y = nearest
            confidence = "nearest-frame"
        estimates.append(
            {
                "pickup_idx": idx,
                "time_s": t,
                "distance": dist,
                "x": x,
                "y": y,
                "confidence": confidence,
            }
        )

    missing = [e for e in estimates if e["confidence"] != "within-radius"]
    if len(missing) == 1 and total_time >= max(float(f["time"]) for f in frames):
        missing[0]["time_s"] = total_time
        missing[0]["confidence"] = "completion-time"

    order = sorted(range(len(estimates)), key=lambda i: estimates[i]["time_s"])
    return estimates, order


def run_trace(
    level: int,
    max_steps: int,
    trace_path: Path,
) -> tuple[list[PickupEvent], list[dict], str, int, bool, list[int]]:
    trace_level, actions = load_action_file(trace_path)
    if trace_level is not None and trace_level != level:
        raise ValueError(f"{trace_path} declares level {trace_level}, expected {level}")

    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    env.reset()
    events, frames, ticks, completed = _run_action_indices(env, actions)
    return events, frames, f"trace:{trace_path}", ticks, completed, actions[:ticks]


def _run_action_indices(
    env: SpaceAceDirectEnv,
    actions: list[int],
) -> tuple[list[PickupEvent], list[dict], int, bool]:
    prev_states = list(env.get_pickup_states())
    events: list[PickupEvent] = []
    frames: list[dict] = []
    ticks = 0
    completed = False
    for action_idx in actions:
        obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
        ticks += 1
        speed = math.hypot(float(obs[2]), float(obs[3]))
        frames.append(
            {
                "time": ticks / 60.0,
                "x": float(obs[0]),
                "y": float(obs[1]),
                "speed": speed,
                "thrusting": int(ALL_ACTIONS[action_idx][2]) > 0,
            }
        )
        states = list(env.get_pickup_states())
        for idx, (before, after) in enumerate(zip(prev_states, states)):
            if not before and after:
                events.append(
                    PickupEvent(
                        pickup_idx=idx,
                        tick=ticks,
                        time_s=ticks / 60.0,
                        x=float(obs[0]),
                        y=float(obs[1]),
                        speed=speed,
                    )
                )
        prev_states = states
        if info.get("level_completed"):
            completed = True
            break
        if terminated or truncated:
            break
    return events, frames, ticks, completed


def run_agent(
    level: int,
    max_steps: int,
    args: argparse.Namespace,
) -> tuple[list[PickupEvent], list[dict], str, int, bool, list[int]]:
    if args.agent not in AGENT_REGISTRY:
        raise ValueError(f"unknown agent {args.agent!r}; available: {', '.join(sorted(AGENT_REGISTRY))}")

    setup_kwargs: dict[str, Any] = {
        "num_simulations": args.num_simulations,
        "exploration_constant": args.exploration,
        "action_repeat": args.action_repeat,
        "thrust_bias": args.thrust_bias,
        "thrust_bias_safe_dist": args.thrust_bias_safe_dist,
        "widen_k": args.widen_k,
        "early_exit_check_every": args.ee_check_every,
        "early_exit_visit_frac": args.ee_visit_frac,
        "early_exit_q_gap": args.ee_q_gap,
    }
    if args.seed is not None:
        spaceace_rl.set_rng_seed(args.seed)
    agent = AGENT_REGISTRY[args.agent]()
    agent.setup(level=level, max_steps=max_steps, **setup_kwargs)
    agent.reset()
    if args.seed is not None:
        spaceace_rl.set_rng_seed(args.seed)

    raw_env = agent.get_raw_env()
    prev_states = list(raw_env.get_pickup_states())
    events: list[PickupEvent] = []
    frames: list[dict] = []
    info: dict[str, Any] = {}
    ticks = 0
    completed = False
    action_indices: list[int] = []
    while True:
        action, _reward, terminated, truncated, info = agent.step()
        obs = raw_env.get_observation()
        tick = int(info.get("step_count", ticks + 1))
        delta_ticks = max(0, tick - ticks)
        if delta_ticks:
            action_indices.extend([action_to_index(action)] * delta_ticks)
        speed = math.hypot(float(obs[2]), float(obs[3]))
        frames.append(
            {
                "time": tick / 60.0,
                "x": float(obs[0]),
                "y": float(obs[1]),
                "speed": speed,
                "thrusting": int(action[2]) > 0,
            }
        )
        states = list(raw_env.get_pickup_states())
        for idx, (before, after) in enumerate(zip(prev_states, states)):
            if not before and after:
                events.append(
                    PickupEvent(
                        pickup_idx=idx,
                        tick=tick,
                        time_s=tick / 60.0,
                        x=float(obs[0]),
                        y=float(obs[1]),
                        speed=speed,
                    )
                )
        prev_states = states
        ticks = tick
        if info.get("level_completed"):
            completed = True
        if terminated or truncated:
            break

    agent.close()
    label = (
        f"{args.agent}(sims={args.num_simulations}, ar={args.action_repeat}, "
        f"tb={args.thrust_bias}, widen={args.widen_k}, seed={args.seed})"
    )
    return events, frames, label, ticks, completed, action_indices[:ticks]


def segment_stats_from_agent_frames(frames: list[dict], t0: float, t1: float) -> SegmentStats:
    sub = [f for f in frames if t0 <= float(f["time"]) <= t1]
    if not sub:
        return SegmentStats(t0, t1, max(0.0, t1 - t0), 0.0, 0.0, 0.0, 0.0)
    speeds = [float(f["speed"]) for f in sub]
    speeds_sorted = sorted(speeds)
    p90 = speeds_sorted[min(len(speeds_sorted) - 1, int(len(speeds_sorted) * 0.9))]
    return SegmentStats(
        t0=t0,
        t1=t1,
        duration=max(0.0, t1 - t0),
        mean_speed=sum(speeds) / len(speeds),
        p90_speed=p90,
        max_speed=max(speeds),
        thrust_fraction=sum(1 for f in sub if f.get("thrusting")) / len(sub),
    )


def print_comparison(
    agent_events: list[PickupEvent],
    agent_frames: list[dict],
    agent_label: str,
    agent_ticks: int,
    agent_completed: bool,
    human_frames: list[dict],
    human_total: float,
    human_estimates: list[dict],
    human_order: list[int],
) -> None:
    agent_order = [e.pickup_idx for e in agent_events]
    print(f"agent: {agent_label}")
    print(
        f"  outcome={'completed' if agent_completed else 'incomplete'} "
        f"time={agent_ticks / 60.0:.2f}s ticks={agent_ticks} order={agent_order}"
    )
    print(f"human: time={human_total:.2f}s estimated_order={human_order}")
    print()

    print("human pickup timing estimates:")
    for e in sorted(human_estimates, key=lambda r: r["time_s"]):
        print(
            f"  P{e['pickup_idx']}: t={e['time_s']:.2f}s "
            f"dist={e['distance']:.1f}px confidence={e['confidence']}"
        )
    print()

    print("rank | pickup | agent seg | human seg | gap | agent speed | human speed | thrust")
    print("-" * 96)
    prev_agent_t = 0.0
    prev_human_t = 0.0
    max_rank = min(len(agent_events), len(human_order))
    for rank in range(max_rank):
        ae = agent_events[rank]
        human_idx = human_order[rank]
        he = human_estimates[human_idx]
        agent_seg = segment_stats_from_agent_frames(agent_frames, prev_agent_t, ae.time_s)
        human_seg = _speed_series(human_frames, prev_human_t, float(he["time_s"]))
        label = f"P{ae.pickup_idx}" if ae.pickup_idx == human_idx else f"P{ae.pickup_idx}/P{human_idx}"
        print(
            f"{rank + 1:>4} | {label:>6} | "
            f"{agent_seg.duration:>8.2f}s | {human_seg.duration:>8.2f}s | "
            f"{agent_seg.duration - human_seg.duration:>+6.2f}s | "
            f"{agent_seg.mean_speed:>5.0f}/{agent_seg.p90_speed:>5.0f}/{agent_seg.max_speed:>5.0f} | "
            f"{human_seg.mean_speed:>5.0f}/{human_seg.p90_speed:>5.0f}/{human_seg.max_speed:>5.0f} | "
            f"{agent_seg.thrust_fraction:>4.0%}/{human_seg.thrust_fraction:>4.0%}"
        )
        prev_agent_t = ae.time_s
        prev_human_t = float(he["time_s"])

    if agent_completed and agent_events:
        tail_agent = agent_ticks / 60.0 - agent_events[-1].time_s
        print(f"\nagent post-final-pickup tail: {tail_agent:.2f}s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--human-ghost", default="human")
    p.add_argument("--human-radius", type=float, default=PICKUP_RADIUS)
    p.add_argument("--trace-json", type=Path, default=None)
    p.add_argument("--agent", default="mcts")
    p.add_argument("--num-simulations", type=int, default=1000)
    p.add_argument("--exploration", type=float, default=1.41)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--thrust-bias", type=float, default=1.0)
    p.add_argument("--thrust-bias-safe-dist", type=float, default=160.0)
    p.add_argument("--widen-k", type=float, default=0.0)
    p.add_argument("--ee-check-every", type=int, default=500)
    p.add_argument("--ee-visit-frac", type=float, default=0.7)
    p.add_argument("--ee-q-gap", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dump-json", type=Path, default=None,
                   help="Write the exact action trace that was compared.")
    args = p.parse_args()

    pickups = pickup_coords(args.level)
    human_frames, human_total = load_ghost(args.level, args.human_ghost)
    human_estimates, human_order = estimate_human_events(
        human_frames, human_total, pickups, args.human_radius
    )

    if args.trace_json is not None:
        agent_events, agent_frames, agent_label, ticks, completed, action_indices = run_trace(
            args.level, args.max_steps, args.trace_json
        )
    else:
        agent_events, agent_frames, agent_label, ticks, completed, action_indices = run_agent(
            args.level, args.max_steps, args
        )

    print_comparison(
        agent_events,
        agent_frames,
        agent_label,
        ticks,
        completed,
        human_frames,
        human_total,
        human_estimates,
        human_order,
    )
    if args.dump_json is not None:
        dump_action_file(args.dump_json, args.level, action_indices, ticks)
        print(f"\ndumped exact action trace: {args.dump_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
