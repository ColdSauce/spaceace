"""Repair a TAS segment by tracking a known-good reference corridor.

This is for cases where a faster prefix reaches a pickup in a slightly different
state, and directly appending the old suffix crashes. The search starts from the
new prefix state, expands exact simulator actions, and ranks live states by how
well they track the reference segment's phase-space corridor. A result is only
accepted when the target pickup is actually collected by the simulator.

Example:
    uv run python scripts/tas_trace_repair.py --level 7 \
      --prefix-json /tmp/spaceace_l7_p2_416_viable.json \
      --reference-json /tmp/spaceace_l7_local_full.json \
      --reference-start 422 --reference-end 738 --target-pickup 1 \
      --dump-prefix-json /tmp/spaceace_l7_p2p1_repaired_prefix.json
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import spaceace_rl  # noqa: E402
from spaceace.core.env import SpaceAceDirectEnv  # noqa: E402
from spaceace.ghost_actions import dump_action_file, load_action_file  # noqa: E402
from spaceace.strategies.actions import ACTION_NAMES, ALL_ACTIONS  # noqa: E402

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class RefPoint:
    obs: tuple[float, float, float, float, float]
    min_wall: float
    action_idx: int


@dataclass(frozen=True)
class Node:
    score: float
    ticks: int
    state: object
    obs: tuple[float, float, float, float, float]
    actions: tuple[int, ...]
    last_action: int


@dataclass(frozen=True)
class RepairResult:
    ticks: int
    actions: list[int]
    final_obs: tuple[float, float, float, float, float]
    score: float


@dataclass(frozen=True)
class RepairSearchResult:
    completed: Optional[RepairResult]
    best_effort: Optional[RepairResult]
    best_effort_distance: float


def _obs5(obs) -> tuple[float, float, float, float, float]:
    return (
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        float(obs[4]),
    )


def _angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % TWO_PI - math.pi


def _load_prefix_state(
    level: int,
    prefix_actions: list[int],
    *,
    max_steps: int,
) -> tuple[object, tuple[float, float, float, float, float], list[bool]]:
    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    env.reset()
    obs = env.get_observation()
    for action_idx in prefix_actions:
        obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
        if info.get("ship_exploded") or terminated or truncated:
            raise ValueError(
                f"prefix terminated before its end at tick {int(info.get('step_count', 0))}"
            )
    return env.save_state(), _obs5(obs), list(env.get_pickup_states())


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


def _reference_points(
    level: int,
    reference_actions: list[int],
    *,
    reference_start: int,
    reference_end: int,
    max_steps: int,
) -> list[RefPoint]:
    if not 0 <= reference_start < reference_end <= len(reference_actions):
        raise ValueError(
            f"reference range {reference_start}..{reference_end} outside trace length "
            f"{len(reference_actions)}"
        )
    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    env.reset()
    points: list[RefPoint] = []
    for tick, action_idx in enumerate(reference_actions[:reference_end], start=1):
        obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
        if tick > reference_start:
            points.append(
                RefPoint(
                    obs=_obs5(obs),
                    min_wall=float(min(obs[8:16])) if len(obs) >= 16 else 1000.0,
                    action_idx=action_idx,
                )
            )
        if info.get("ship_exploded") or terminated or truncated:
            break
    if len(points) != reference_end - reference_start:
        raise ValueError("reference segment terminated before requested end")
    return points


def _score(
    obs: tuple[float, float, float, float, float],
    ref: RefPoint,
    *,
    target_xy: Optional[tuple[float, float]],
    action_idx: int,
    last_action: int,
    tick_cost: float,
    ticks: int,
    action_mismatch_cost: float,
    switch_cost: float,
    pickup_weight: float,
) -> float:
    x, y, vx, vy, rot = obs
    rx, ry, rvx, rvy, rrot = ref.obs
    pos = math.hypot(x - rx, y - ry)
    vel = math.hypot(vx - rvx, vy - rvy)
    rot_err = abs(_angle_diff(rot, rrot))
    action_mismatch = 0.0 if action_idx == ref.action_idx else action_mismatch_cost
    switch = 0.0 if action_idx == last_action else switch_cost
    clearance_penalty = 0.0
    if ref.min_wall > 0:
        # If the reference corridor is tight, discourage getting materially
        # closer to walls than the successful trace at this phase.
        clearance_penalty = 0.0
    pickup_score = 0.0
    if target_xy is not None and pickup_weight > 0.0:
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
        heading = abs(_angle_diff(rot, desired_rot))
        pickup_score = pickup_weight * (
            dist
            - 0.45 * toward
            + 0.08 * lateral
            + 22.0 * heading
        )
    return (
        pos
        + 0.45 * vel
        + 45.0 * rot_err
        + action_mismatch
        + switch
        + clearance_penalty
        + pickup_score
        + tick_cost * ticks
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
        + rot_weight * abs(_angle_diff(rot, rrot))
    )


def _quantize(
    obs: tuple[float, float, float, float, float],
    *,
    pos_q: float,
    vel_q: float,
    rot_q: float,
) -> tuple[int, int, int, int, int]:
    x, y, vx, vy, rot = obs
    return (
        int(round(x / pos_q)),
        int(round(y / pos_q)),
        int(round(vx / vel_q)),
        int(round(vy / vel_q)),
        int(round((rot % TWO_PI) / rot_q)),
    )


def _action_set(ref_action: int, *, all_actions: bool) -> Iterable[int]:
    if all_actions:
        return range(len(ALL_ACTIONS))
    # Prefer exact reference action and small control neighbors. Include all
    # thrust/coast variants for the same rotation direction and both pure turns.
    raw = tuple(int(x) for x in ALL_ACTIONS[ref_action].tolist())
    left, right, thrust = raw
    choices = {ref_action}
    choices.add(0 if thrust else 1)  # coast/thrust straight counterpart
    if left:
        choices.update({2, 3})
    elif right:
        choices.update({4, 5})
    else:
        choices.update({0, 1, 2, 4})
    return sorted(choices)


def repair_segment(
    *,
    level: int,
    prefix_actions: list[int],
    reference_actions: list[int],
    reference_start: int,
    reference_end: int,
    target_pickup: int,
    max_steps: int,
    max_extra_ticks: int,
    beam_width: int,
    keep_per_bucket: int,
    pos_q: float,
    vel_q: float,
    rot_q: float,
    tick_cost: float,
    action_mismatch_cost: float,
    switch_cost: float,
    all_actions: bool,
    pickup_bias_start: int,
    pickup_bias_ramp: int,
    pickup_bias_weight: float,
    continue_after_reach: bool,
    arrival_ref_obs: Optional[tuple[float, float, float, float, float]],
    arrival_tick_weight: float,
    arrival_pos_weight: float,
    arrival_vel_weight: float,
    arrival_rot_weight: float,
) -> RepairSearchResult:
    ref_points = _reference_points(
        level,
        reference_actions,
        reference_start=reference_start,
        reference_end=reference_end,
        max_steps=max_steps,
    )
    start_state, start_obs, start_pickups = _load_prefix_state(
        level,
        prefix_actions,
        max_steps=max_steps,
    )
    if start_pickups[target_pickup]:
        result = RepairResult(0, [], start_obs, 0.0)
        return RepairSearchResult(result, result, 0.0)

    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    pf = spaceace_rl.PyPathfinder(level, "grid")
    pickup_coords = [(float(x), float(y)) for x, y in pf.get_pickup_coords()]
    target_xy = pickup_coords[target_pickup]
    last_prefix_action = prefix_actions[-1] if prefix_actions else 0
    start_ref = ref_points[0]
    start_score = _score(
        start_obs,
        start_ref,
        target_xy=None,
        action_idx=last_prefix_action,
        last_action=last_prefix_action,
        tick_cost=tick_cost,
        ticks=0,
        action_mismatch_cost=action_mismatch_cost,
        switch_cost=switch_cost,
        pickup_weight=0.0,
    )
    beam = [Node(start_score, 0, start_state, start_obs, (), last_prefix_action)]
    best: Optional[RepairResult] = None
    best_effort: Optional[RepairResult] = None
    best_effort_distance = math.inf
    max_ticks = len(ref_points) + max_extra_ticks
    t0 = time.time()

    print(
        f"repair start={tuple(round(v, 2) for v in start_obs)} "
        f"reference_ticks={len(ref_points)} max_ticks={max_ticks}"
    )

    for tick in range(1, max_ticks + 1):
        ref_idx = min(len(ref_points) - 1, tick - 1)
        ref = ref_points[ref_idx]
        if pickup_bias_weight > 0.0 and tick >= pickup_bias_start:
            ramp = max(1, pickup_bias_ramp)
            pickup_weight = pickup_bias_weight * min(1.0, (tick - pickup_bias_start + 1) / ramp)
            score_target = target_xy
        else:
            pickup_weight = 0.0
            score_target = None
        expanded: list[Node] = []
        bucket_counts: dict[tuple[int, int, int, int, int], int] = {}
        reached = 0
        expansions = 0

        for node in beam:
            for action_idx in _action_set(ref.action_idx, all_actions=all_actions):
                env.load_state(node.state)
                obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
                expansions += 1
                if info.get("ship_exploded") or terminated or truncated:
                    continue

                obs5 = _obs5(obs)
                actions = node.actions + (action_idx,)
                states = list(env.get_pickup_states())
                dist_to_target = math.hypot(obs5[0] - target_xy[0], obs5[1] - target_xy[1])
                if dist_to_target < best_effort_distance:
                    best_effort_distance = dist_to_target
                    best_effort = RepairResult(
                        ticks=tick,
                        actions=list(actions),
                        final_obs=obs5,
                        score=0.0,
                    )
                if states[target_pickup]:
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
                    result = RepairResult(
                        ticks=tick,
                        actions=list(actions),
                        final_obs=obs5,
                        score=completion_score,
                    )
                    if (
                        best is None
                        or result.score < best.score
                        or (
                            math.isclose(result.score, best.score)
                            and result.ticks < best.ticks
                        )
                    ):
                        best = result
                        names = " ".join(ACTION_NAMES[a] for a in result.actions[:12])
                        print(
                            f"  reached P{target_pickup}: {result.ticks}t "
                            f"score={result.score:.1f} "
                            f"first=[{names}]"
                        )
                    continue

                if best is not None and not continue_after_reach and tick >= best.ticks:
                    continue

                key = _quantize(obs5, pos_q=pos_q, vel_q=vel_q, rot_q=rot_q)
                count = bucket_counts.get(key, 0)
                if count >= keep_per_bucket:
                    continue
                bucket_counts[key] = count + 1

                score = _score(
                    obs5,
                    ref,
                    target_xy=score_target,
                    action_idx=action_idx,
                    last_action=node.last_action,
                    tick_cost=tick_cost,
                    ticks=tick,
                    action_mismatch_cost=action_mismatch_cost,
                    switch_cost=switch_cost,
                    pickup_weight=pickup_weight,
                )
                expanded.append(Node(score, tick, env.save_state(), obs5, actions, action_idx))

        if expanded:
            beam = heapq.nsmallest(beam_width, expanded, key=lambda n: n.score)
        else:
            beam = []

        if tick % 20 == 0 or reached or not beam:
            best_text = f"{best.ticks}t" if best is not None else "-"
            score_text = f"{beam[0].score:.1f}" if beam else "-"
            dist_text = f"{best_effort_distance:.1f}" if math.isfinite(best_effort_distance) else "-"
            print(
                f"tick={tick} expanded={expansions} live={len(beam)} "
                f"reached={reached} best={best_text} best_dist={dist_text} score={score_text} "
                f"elapsed={time.time() - t0:.1f}s"
            )

        if best is not None and not continue_after_reach:
            break
        if not beam:
            break

    return RepairSearchResult(best, best_effort, best_effort_distance)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--prefix-json", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--reference-start", type=int, required=True)
    parser.add_argument("--reference-end", type=int, required=True)
    parser.add_argument("--target-pickup", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--max-extra-ticks", type=int, default=30)
    parser.add_argument("--beam-width", type=int, default=4000)
    parser.add_argument("--keep-per-bucket", type=int, default=3)
    parser.add_argument("--pos-q", type=float, default=8.0)
    parser.add_argument("--vel-q", type=float, default=18.0)
    parser.add_argument("--rot-q-deg", type=float, default=5.0)
    parser.add_argument("--tick-cost", type=float, default=0.05)
    parser.add_argument("--action-mismatch-cost", type=float, default=1.0)
    parser.add_argument("--switch-cost", type=float, default=0.15)
    parser.add_argument(
        "--pickup-bias-start",
        type=int,
        default=0,
        help="Start blending target-pickup capture into the score after this segment tick.",
    )
    parser.add_argument(
        "--pickup-bias-ramp",
        type=int,
        default=80,
        help="Ticks over which pickup capture score ramps to full strength.",
    )
    parser.add_argument(
        "--pickup-bias-weight",
        type=float,
        default=0.0,
        help="Weight of final pickup-capture score; 0 disables it.",
    )
    parser.add_argument("--continue-after-reach", action="store_true")
    parser.add_argument("--arrival-reference-json", type=Path, default=None)
    parser.add_argument("--arrival-reference-tick", type=int, default=None)
    parser.add_argument("--arrival-tick-weight", type=float, default=1.0)
    parser.add_argument("--arrival-pos-weight", type=float, default=0.0)
    parser.add_argument("--arrival-vel-weight", type=float, default=0.0)
    parser.add_argument("--arrival-rot-weight", type=float, default=0.0)
    parser.add_argument("--all-actions", action="store_true")
    parser.add_argument("--dump-segment-json", type=Path, default=None)
    parser.add_argument("--dump-prefix-json", type=Path, default=None)
    parser.add_argument("--dump-best-effort-segment-json", type=Path, default=None)
    parser.add_argument("--dump-best-effort-prefix-json", type=Path, default=None)
    args = parser.parse_args()

    prefix_level, prefix_actions = load_action_file(args.prefix_json)
    reference_level, reference_actions = load_action_file(args.reference_json)
    for path, declared in ((args.prefix_json, prefix_level), (args.reference_json, reference_level)):
        if declared is not None and declared != args.level:
            raise ValueError(f"{path} declares level {declared}, expected {args.level}")
    arrival_ref_obs = None
    if args.arrival_reference_json is not None:
        if args.arrival_reference_tick is None:
            raise ValueError("--arrival-reference-tick is required with --arrival-reference-json")
        arrival_level, arrival_actions = load_action_file(args.arrival_reference_json)
        if arrival_level is not None and arrival_level != args.level:
            raise ValueError(
                f"{args.arrival_reference_json} declares level {arrival_level}, expected {args.level}"
            )
        arrival_ref_obs = _replay_to_tick(
            args.level,
            arrival_actions,
            args.arrival_reference_tick,
            max_steps=args.max_steps,
        )
        print(
            "arrival reference "
            f"tick={args.arrival_reference_tick} "
            f"obs={tuple(round(v, 2) for v in arrival_ref_obs)}"
        )

    search_result = repair_segment(
        level=args.level,
        prefix_actions=prefix_actions,
        reference_actions=reference_actions,
        reference_start=args.reference_start,
        reference_end=args.reference_end,
        target_pickup=args.target_pickup,
        max_steps=args.max_steps,
        max_extra_ticks=args.max_extra_ticks,
        beam_width=args.beam_width,
        keep_per_bucket=args.keep_per_bucket,
        pos_q=args.pos_q,
        vel_q=args.vel_q,
        rot_q=math.radians(args.rot_q_deg),
        tick_cost=args.tick_cost,
        action_mismatch_cost=args.action_mismatch_cost,
        switch_cost=args.switch_cost,
        all_actions=args.all_actions,
        pickup_bias_start=args.pickup_bias_start,
        pickup_bias_ramp=args.pickup_bias_ramp,
        pickup_bias_weight=args.pickup_bias_weight,
        continue_after_reach=args.continue_after_reach,
        arrival_ref_obs=arrival_ref_obs,
        arrival_tick_weight=args.arrival_tick_weight,
        arrival_pos_weight=args.arrival_pos_weight,
        arrival_vel_weight=args.arrival_vel_weight,
        arrival_rot_weight=args.arrival_rot_weight,
    )
    result = search_result.completed
    if result is None:
        if search_result.best_effort is not None:
            print(
                f"best effort: {search_result.best_effort.ticks}t "
                f"dist={search_result.best_effort_distance:.1f}px "
                f"obs={[round(v, 3) for v in search_result.best_effort.final_obs]}"
            )
            if args.dump_best_effort_segment_json is not None:
                dump_action_file(
                    args.dump_best_effort_segment_json,
                    args.level,
                    search_result.best_effort.actions,
                    search_result.best_effort.ticks,
                )
                print(f"dumped best-effort segment -> {args.dump_best_effort_segment_json}")
            if args.dump_best_effort_prefix_json is not None:
                full_prefix = prefix_actions + search_result.best_effort.actions
                dump_action_file(
                    args.dump_best_effort_prefix_json,
                    args.level,
                    full_prefix,
                    len(full_prefix),
                )
                print(f"dumped best-effort prefix -> {args.dump_best_effort_prefix_json}")
        print("no repair found")
        return 1

    if args.dump_segment_json is not None:
        dump_action_file(args.dump_segment_json, args.level, result.actions, result.ticks)
        print(f"dumped segment -> {args.dump_segment_json}")
    if args.dump_prefix_json is not None:
        full_prefix = prefix_actions + result.actions
        dump_action_file(args.dump_prefix_json, args.level, full_prefix, len(full_prefix))
        print(f"dumped prefix -> {args.dump_prefix_json}")

    print(json.dumps({
        "level": args.level,
        "target_pickup": args.target_pickup,
        "segment_ticks": result.ticks,
        "completion_score": result.score,
        "prefix_ticks": len(prefix_actions),
        "total_ticks": len(prefix_actions) + result.ticks,
        "final_obs": [round(v, 3) for v in result.final_obs],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
