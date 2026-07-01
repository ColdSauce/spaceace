"""Repair a TAS segment against sampled human ghost frames.

The dashboard human ghost is not an exact action log, but it is a useful
phase-space reference for racing lines. This script starts from an exact TAS
prefix state, expands exact simulator actions, and ranks states by proximity to
the sampled human trajectory for a chosen pickup-to-pickup segment. A result is
accepted only when the requested target pickup is actually collected.
"""

from __future__ import annotations

import argparse
import bisect
import heapq
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import spaceace_rl  # noqa: E402
from spaceace.core.env import SpaceAceDirectEnv  # noqa: E402
from spaceace.ghost_actions import dump_action_file, load_action_file  # noqa: E402
from spaceace.strategies.actions import ACTION_NAMES, ALL_ACTIONS  # noqa: E402
from scripts.compare_agent_human_segments import (  # noqa: E402
    PICKUP_RADIUS,
    estimate_human_events,
    load_ghost,
    pickup_coords,
)

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class FrameRef:
    x: float
    y: float
    vx: float
    vy: float
    rot: float
    thrusting: bool


@dataclass(frozen=True)
class Node:
    score: float
    ticks: int
    state: object
    actions: tuple[int, ...]
    last_action: int


@dataclass(frozen=True)
class SearchResult:
    completed_actions: Optional[list[int]]
    completed_ticks: Optional[int]
    completed_score: Optional[float]
    best_effort_actions: Optional[list[int]]
    best_effort_ticks: int
    best_effort_distance: float
    best_effort_obs: Optional[tuple[float, float, float, float, float]]


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % TWO_PI - math.pi


def _obs5(obs) -> tuple[float, float, float, float, float]:
    return (
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        float(obs[4]),
    )


def _load_prefix_state(
    level: int,
    prefix_actions: list[int],
    *,
    max_steps: int,
) -> tuple[object, tuple[float, float, float, float, float]]:
    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    env.reset()
    obs = env.get_observation()
    for action_idx in prefix_actions:
        obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
        if info.get("ship_exploded") or terminated or truncated:
            raise ValueError(
                f"prefix terminated before its end at tick {int(info.get('step_count', 0))}"
            )
    return env.save_state(), _obs5(obs)


def _replay_to_tick(
    level: int,
    actions: list[int],
    tick: int,
    *,
    max_steps: int,
) -> tuple[float, float, float, float, float]:
    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    env.reset()
    obs = env.get_observation()
    if not 0 <= tick <= len(actions):
        raise ValueError(f"arrival reference tick {tick} outside trace length {len(actions)}")
    for action_idx in actions[:tick]:
        obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
        if info.get("ship_exploded") or terminated or truncated:
            raise ValueError(f"arrival reference terminated before tick {tick}")
    return _obs5(obs)


def _unique_human_frames(level: int, human_ghost: str) -> tuple[list[dict], float, list[dict], list[int]]:
    frames, total_time = load_ghost(level, human_ghost)
    pickups = pickup_coords(level)
    estimates, order = estimate_human_events(frames, total_time, pickups, PICKUP_RADIUS)

    # Dashboard ghosts can contain duplicate timestamps. Keep the last frame for
    # each rounded physics tick to get a monotone interpolation domain.
    by_tick: dict[int, dict] = {}
    for frame in frames:
        tick = int(round(float(frame["time"]) * 60.0))
        by_tick[tick] = frame
    unique = []
    for tick in sorted(by_tick):
        frame = dict(by_tick[tick])
        frame["_tick"] = tick
        unique.append(frame)
    return unique, total_time, estimates, order


def _interp_angle(a: float, b: float, u: float) -> float:
    return a + _wrap_to_pi(b - a) * u


def _make_frame_reference(
    *,
    level: int,
    human_ghost: str,
    start_pickup: Optional[int],
    target_pickup: Optional[int],
    start_time_s: Optional[float],
    target_time_s: Optional[float],
    max_ticks: int,
    speed_scale: float,
) -> tuple[list[FrameRef], float]:
    frames, total_time, estimates, _order = _unique_human_frames(level, human_ghost)
    if start_time_s is not None:
        start_t = float(start_time_s)
    elif start_pickup is not None:
        start_t = float(estimates[start_pickup]["time_s"])
    else:
        start_t = 0.0

    if target_time_s is not None:
        target_t = float(target_time_s)
    elif target_pickup is not None:
        target_t = float(estimates[target_pickup]["time_s"])
    else:
        target_t = total_time

    if target_t <= start_t:
        target_t = total_time
    segment_seconds = target_t - start_t
    segment_ticks = max(1, int(round(segment_seconds * 60.0 / speed_scale)))

    src_ticks = [int(f["_tick"]) for f in frames]
    refs: list[FrameRef] = []
    for tick in range(1, max_ticks + 1):
        human_abs_tick = int(round(start_t * 60.0 + tick * speed_scale))
        j = bisect.bisect_left(src_ticks, human_abs_tick)
        if j <= 0:
            f0 = f1 = frames[0]
            u = 0.0
        elif j >= len(frames):
            f0 = f1 = frames[-1]
            u = 0.0
        else:
            f0 = frames[j - 1]
            f1 = frames[j]
            dt = max(1, int(f1["_tick"]) - int(f0["_tick"]))
            u = (human_abs_tick - int(f0["_tick"])) / dt
            u = max(0.0, min(1.0, u))

        x0, y0 = float(f0["x"]), float(f0["y"])
        x1, y1 = float(f1["x"]), float(f1["y"])
        x = x0 + (x1 - x0) * u
        y = y0 + (y1 - y0) * u
        rot = _interp_angle(float(f0["rotation"]), float(f1["rotation"]), u)
        dt_seconds = max(1e-6, (float(f1["time"]) - float(f0["time"])))
        vx = (x1 - x0) / dt_seconds if f0 is not f1 else 0.0
        vy = (y1 - y0) / dt_seconds if f0 is not f1 else 0.0
        refs.append(FrameRef(x, y, vx, vy, rot, bool(f0.get("thrusting"))))
    return refs, segment_ticks


def _score(
    obs,
    ref: FrameRef,
    *,
    target_xy: tuple[float, float],
    tick: int,
    action_idx: int,
    last_action: int,
    pickup_weight: float,
    tick_cost: float,
    switch_cost: float,
    pos_weight: float,
    vel_weight: float,
    rot_weight: float,
) -> float:
    x, y, vx, vy, rot = _obs5(obs)
    pos = math.hypot(x - ref.x, y - ref.y)
    vel = math.hypot(vx - ref.vx, vy - ref.vy)
    rot_err = abs(_wrap_to_pi(rot - ref.rot))
    tx, ty = target_xy
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    if dist > 1e-6:
        ux = dx / dist
        uy = dy / dist
    else:
        ux = uy = 0.0
    toward = vx * ux + vy * uy
    lateral = abs(vx * -uy + vy * ux)
    desired_rot = math.atan2(uy, ux) + math.pi * 0.5
    heading = abs(_wrap_to_pi(rot - desired_rot))
    action_switch = switch_cost if action_idx != last_action else 0.0
    return (
        pos_weight * pos
        + vel_weight * vel
        + rot_weight * rot_err
        + pickup_weight * (dist - 0.35 * toward + 0.05 * lateral + 14.0 * heading)
        + action_switch
        + tick_cost * tick
    )


def _arrival_score(
    obs: tuple[float, float, float, float, float],
    ticks: int,
    *,
    ref_obs: Optional[tuple[float, float, float, float, float]],
    tick_weight: float,
    pos_weight: float,
    vel_weight: float,
    rot_weight: float,
) -> float:
    if ref_obs is None:
        return float(ticks)
    x, y, vx, vy, rot = obs
    rx, ry, rvx, rvy, rrot = ref_obs
    return (
        tick_weight * ticks
        + pos_weight * math.hypot(x - rx, y - ry)
        + vel_weight * math.hypot(vx - rvx, vy - rvy)
        + rot_weight * abs(_wrap_to_pi(rot - rrot))
    )


def _quantize(obs, *, pos_q: float, vel_q: float, rot_q: float) -> tuple[int, int, int, int, int]:
    x, y, vx, vy, rot = _obs5(obs)
    return (
        int(round(x / pos_q)),
        int(round(y / pos_q)),
        int(round(vx / vel_q)),
        int(round(vy / vel_q)),
        int(round((rot % TWO_PI) / rot_q)),
    )


def search_human_repair(
    *,
    level: int,
    prefix_actions: list[int],
    human_ghost: str,
    start_pickup: Optional[int],
    target_pickup: int,
    start_time_s: Optional[float],
    target_time_s: Optional[float],
    max_ticks: int,
    speed_scale: float,
    beam_width: int,
    keep_per_bucket: int,
    pos_q: float,
    vel_q: float,
    rot_q: float,
    pickup_weight: float,
    tick_cost: float,
    switch_cost: float,
    pos_weight: float,
    vel_weight: float,
    rot_weight: float,
    continue_after_reach: bool,
    arrival_ref_obs: Optional[tuple[float, float, float, float, float]],
    arrival_tick_weight: float,
    arrival_pos_weight: float,
    arrival_vel_weight: float,
    arrival_rot_weight: float,
    max_steps: int,
) -> SearchResult:
    refs, human_segment_ticks = _make_frame_reference(
        level=level,
        human_ghost=human_ghost,
        start_pickup=start_pickup,
        target_pickup=target_pickup,
        start_time_s=start_time_s,
        target_time_s=target_time_s,
        max_ticks=max_ticks,
        speed_scale=speed_scale,
    )
    pf = spaceace_rl.PyPathfinder(level, "grid")
    target_xy = tuple(float(v) for v in pf.get_pickup_coords()[target_pickup])
    start_state, start_obs = _load_prefix_state(level, prefix_actions, max_steps=max_steps)

    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    last_action = prefix_actions[-1] if prefix_actions else 0
    beam = [Node(0.0, 0, start_state, (), last_action)]
    best_effort_actions: Optional[list[int]] = None
    best_effort_obs: Optional[tuple[float, float, float, float, float]] = None
    best_effort_distance = math.inf
    best_effort_ticks = 0
    best_completed_actions: Optional[list[int]] = None
    best_completed_ticks: Optional[int] = None
    best_completed_score: Optional[float] = None
    t0 = time.time()

    print(
        f"human repair start={tuple(round(v, 2) for v in start_obs)} "
        f"human_segment_ticks~{human_segment_ticks:.1f} max_ticks={max_ticks} "
        f"speed_scale={speed_scale}"
    )

    for tick in range(1, max_ticks + 1):
        ref = refs[min(len(refs) - 1, tick - 1)]
        expanded: list[Node] = []
        bucket_counts: dict[tuple[int, int, int, int, int], int] = {}
        reached = 0
        expansions = 0
        for node in beam:
            for action_idx in range(len(ALL_ACTIONS)):
                env.load_state(node.state)
                obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
                expansions += 1
                if info.get("ship_exploded") or terminated or truncated:
                    continue
                obs5 = _obs5(obs)
                actions = node.actions + (action_idx,)
                dist = math.hypot(obs5[0] - target_xy[0], obs5[1] - target_xy[1])
                if dist < best_effort_distance:
                    best_effort_distance = dist
                    best_effort_actions = list(actions)
                    best_effort_obs = obs5
                    best_effort_ticks = tick
                if list(env.get_pickup_states())[target_pickup]:
                    reached += 1
                    completion_score = _arrival_score(
                        obs5,
                        tick,
                        ref_obs=arrival_ref_obs,
                        tick_weight=arrival_tick_weight,
                        pos_weight=arrival_pos_weight,
                        vel_weight=arrival_vel_weight,
                        rot_weight=arrival_rot_weight,
                    )
                    improved = (
                        best_completed_score is None
                        or completion_score < best_completed_score
                        or (
                            math.isclose(completion_score, best_completed_score)
                            and best_completed_ticks is not None
                            and tick < best_completed_ticks
                        )
                    )
                    if improved:
                        print(
                            f"  reached P{target_pickup}: {tick}t "
                            f"score={completion_score:.1f} "
                            f"first={[ACTION_NAMES[a] for a in actions[:10]]}"
                        )
                    if (
                        improved
                    ):
                        best_completed_actions = list(actions)
                        best_completed_ticks = tick
                        best_completed_score = completion_score
                    if not continue_after_reach:
                        return SearchResult(
                            list(actions),
                            tick,
                            completion_score,
                            best_effort_actions,
                            best_effort_ticks,
                            best_effort_distance,
                            best_effort_obs,
                        )
                    continue

                key = _quantize(obs, pos_q=pos_q, vel_q=vel_q, rot_q=rot_q)
                count = bucket_counts.get(key, 0)
                if count >= keep_per_bucket:
                    continue
                bucket_counts[key] = count + 1
                score = _score(
                    obs,
                    ref,
                    target_xy=target_xy,
                    tick=tick,
                    action_idx=action_idx,
                    last_action=node.last_action,
                    pickup_weight=pickup_weight,
                    tick_cost=tick_cost,
                    switch_cost=switch_cost,
                    pos_weight=pos_weight,
                    vel_weight=vel_weight,
                    rot_weight=rot_weight,
                )
                expanded.append(Node(score, tick, env.save_state(), actions, action_idx))

        if expanded:
            beam = heapq.nsmallest(beam_width, expanded, key=lambda n: n.score)
        else:
            beam = []
        if tick % 20 == 0 or reached or not beam:
            print(
                f"tick={tick} expanded={expansions} live={len(beam)} "
                f"best_dist={best_effort_distance:.1f} "
                f"score={(beam[0].score if beam else float('nan')):.1f} "
                f"elapsed={time.time() - t0:.1f}s"
            )
        if not beam:
            break

    return SearchResult(
        best_completed_actions,
        best_completed_ticks,
        best_completed_score,
        best_effort_actions,
        best_effort_ticks,
        best_effort_distance,
        best_effort_obs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--prefix-json", type=Path, required=True)
    parser.add_argument("--human-ghost", default="human")
    parser.add_argument("--start-pickup", type=int, default=None)
    parser.add_argument("--target-pickup", type=int, required=True)
    parser.add_argument("--start-time-s", type=float, default=None)
    parser.add_argument("--target-time-s", type=float, default=None)
    parser.add_argument("--max-ticks", type=int, default=640)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--beam-width", type=int, default=5000)
    parser.add_argument("--keep-per-bucket", type=int, default=4)
    parser.add_argument("--pos-q", type=float, default=12.0)
    parser.add_argument("--vel-q", type=float, default=28.0)
    parser.add_argument("--rot-q-deg", type=float, default=8.0)
    parser.add_argument("--pickup-weight", type=float, default=0.55)
    parser.add_argument("--tick-cost", type=float, default=0.03)
    parser.add_argument("--switch-cost", type=float, default=0.05)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--vel-weight", type=float, default=0.18)
    parser.add_argument("--rot-weight", type=float, default=20.0)
    parser.add_argument("--continue-after-reach", action="store_true")
    parser.add_argument("--arrival-reference-json", type=Path, default=None)
    parser.add_argument("--arrival-reference-tick", type=int, default=None)
    parser.add_argument("--arrival-tick-weight", type=float, default=1.0)
    parser.add_argument("--arrival-pos-weight", type=float, default=0.0)
    parser.add_argument("--arrival-vel-weight", type=float, default=0.0)
    parser.add_argument("--arrival-rot-weight", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--dump-segment-json", type=Path, default=None)
    parser.add_argument("--dump-prefix-json", type=Path, default=None)
    parser.add_argument("--dump-best-effort-segment-json", type=Path, default=None)
    parser.add_argument("--dump-best-effort-prefix-json", type=Path, default=None)
    args = parser.parse_args()

    trace_level, prefix_actions = load_action_file(args.prefix_json)
    if trace_level is not None and trace_level != args.level:
        raise ValueError(f"{args.prefix_json} declares level {trace_level}, expected {args.level}")
    arrival_ref_obs = None
    if args.arrival_reference_json is not None:
        if args.arrival_reference_tick is None:
            raise ValueError("--arrival-reference-tick is required with --arrival-reference-json")
        ref_level, ref_actions = load_action_file(args.arrival_reference_json)
        if ref_level is not None and ref_level != args.level:
            raise ValueError(
                f"{args.arrival_reference_json} declares level {ref_level}, expected {args.level}"
            )
        arrival_ref_obs = _replay_to_tick(
            args.level,
            ref_actions,
            args.arrival_reference_tick,
            max_steps=args.max_steps,
        )
        print(
            "arrival reference "
            f"tick={args.arrival_reference_tick} "
            f"obs={tuple(round(v, 2) for v in arrival_ref_obs)}"
        )

    result = search_human_repair(
        level=args.level,
        prefix_actions=prefix_actions,
        human_ghost=args.human_ghost,
        start_pickup=args.start_pickup,
        target_pickup=args.target_pickup,
        start_time_s=args.start_time_s,
        target_time_s=args.target_time_s,
        max_ticks=args.max_ticks,
        speed_scale=args.speed_scale,
        beam_width=args.beam_width,
        keep_per_bucket=args.keep_per_bucket,
        pos_q=args.pos_q,
        vel_q=args.vel_q,
        rot_q=math.radians(args.rot_q_deg),
        pickup_weight=args.pickup_weight,
        tick_cost=args.tick_cost,
        switch_cost=args.switch_cost,
        pos_weight=args.pos_weight,
        vel_weight=args.vel_weight,
        rot_weight=args.rot_weight,
        continue_after_reach=args.continue_after_reach,
        arrival_ref_obs=arrival_ref_obs,
        arrival_tick_weight=args.arrival_tick_weight,
        arrival_pos_weight=args.arrival_pos_weight,
        arrival_vel_weight=args.arrival_vel_weight,
        arrival_rot_weight=args.arrival_rot_weight,
        max_steps=args.max_steps,
    )

    if result.completed_actions is not None and result.completed_ticks is not None:
        if args.dump_segment_json is not None:
            dump_action_file(args.dump_segment_json, args.level, result.completed_actions, result.completed_ticks)
            print(f"dumped segment -> {args.dump_segment_json}")
        if args.dump_prefix_json is not None:
            full_prefix = prefix_actions + result.completed_actions
            dump_action_file(args.dump_prefix_json, args.level, full_prefix, len(full_prefix))
            print(f"dumped prefix -> {args.dump_prefix_json}")
        print(json.dumps({
            "level": args.level,
            "target_pickup": args.target_pickup,
            "segment_ticks": result.completed_ticks,
            "completion_score": result.completed_score,
            "prefix_ticks": len(prefix_actions),
            "total_ticks": len(prefix_actions) + result.completed_ticks,
        }, indent=2))
        return 0

    print(
        f"best effort: {result.best_effort_ticks}t dist={result.best_effort_distance:.1f}px "
        f"obs={None if result.best_effort_obs is None else [round(v, 3) for v in result.best_effort_obs]}"
    )
    if result.best_effort_actions is not None:
        if args.dump_best_effort_segment_json is not None:
            dump_action_file(
                args.dump_best_effort_segment_json,
                args.level,
                result.best_effort_actions,
                result.best_effort_ticks,
            )
            print(f"dumped best-effort segment -> {args.dump_best_effort_segment_json}")
        if args.dump_best_effort_prefix_json is not None:
            full_prefix = prefix_actions + result.best_effort_actions
            dump_action_file(
                args.dump_best_effort_prefix_json,
                args.level,
                full_prefix,
                len(full_prefix),
            )
            print(f"dumped best-effort prefix -> {args.dump_best_effort_prefix_json}")
    print("no repair found")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
