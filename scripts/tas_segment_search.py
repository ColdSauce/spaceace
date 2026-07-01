"""Exact simulator segment search for TAS traces.

This searches from the end of an exact prefix until a requested pickup is
actually collected by the Rust engine. The queue score is only a way to choose
which exact simulator states to expand under a finite budget; candidates are
accepted only when the target pickup bit flips during replay.

Typical level-7 workflow:
    uv run python scripts/tas_segment_search.py --level 7 \
      --prefix-json /tmp/spaceace_l7_p2_local.json \
      --target-pickup 1 --primitive-ticks 3 --max-depth 130 \
      --beam-width 2500 --dump-prefix-json /tmp/spaceace_l7_p2p1_prefix.json
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
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import spaceace_rl  # noqa: E402
from spaceace.core.env import SpaceAceDirectEnv  # noqa: E402
from spaceace.ghost_actions import dump_action_file, load_action_file  # noqa: E402
from spaceace.strategies.actions import ACTION_NAMES, ALL_ACTIONS  # noqa: E402


@dataclass(frozen=True)
class Node:
    score: float
    ticks: int
    state: object
    obs: tuple[float, float, float, float, float]
    min_wall: float
    pickup_bits: int
    actions: tuple[int, ...]


@dataclass(frozen=True)
class SegmentResult:
    ticks: int
    actions: list[int]
    score: float
    final_obs: tuple[float, float, float, float, float]


TWO_PI = 2.0 * math.pi


def _obs5(obs) -> tuple[float, float, float, float, float]:
    return (
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        float(obs[4]),
    )


def _pickup_bits(states: list[bool]) -> int:
    bits = 0
    for i, collected in enumerate(states):
        if collected:
            bits |= 1 << i
    return bits


def _angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % TWO_PI - math.pi


def _score_to_pickup(
    obs: tuple[float, float, float, float, float],
    target: tuple[float, float],
    *,
    route: Optional[tuple[float, float, float]],
    tick_cost: float,
    ticks: int,
    min_wall: Optional[float],
) -> float:
    """Rank a simulator state for target-pickup search.

    The score is not the objective. It combines distance, velocity projection,
    heading error, and a soft wall-clearance term so the finite beam spends
    expansions on states that are likely to reach the target soon.
    """
    x, y, vx, vy, rot = obs
    tx, ty = target
    if route is not None and math.isfinite(route[0]) and route[0] > 0.0:
        dist = float(route[0])
        ux = float(route[1])
        uy = float(route[2])
    else:
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

    wall_penalty = 0.0
    if min_wall is not None and min_wall < 160.0:
        wall_penalty = (160.0 - min_wall) * 8.0

    return (
        dist
        - 0.65 * toward
        + 0.15 * lateral
        + 35.0 * heading
        + wall_penalty
        + tick_cost * ticks
    )


def _quantize(
    obs: tuple[float, float, float, float, float],
    pickup_bits: int,
    *,
    pos_q: float,
    vel_q: float,
    rot_q: float,
) -> tuple[int, int, int, int, int, int]:
    x, y, vx, vy, rot = obs
    return (
        int(round(x / pos_q)),
        int(round(y / pos_q)),
        int(round(vx / vel_q)),
        int(round(vy / vel_q)),
        int(round((rot % TWO_PI) / rot_q)),
        pickup_bits,
    )


def _load_prefix_state(
    level: int,
    prefix_actions: list[int],
    *,
    max_steps: int,
) -> tuple[object, tuple[float, float, float, float, float], float, int]:
    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    env.reset()
    obs = env.get_observation()
    for action_idx in prefix_actions:
        obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
        if info.get("ship_exploded") or terminated or truncated:
            raise ValueError(
                f"prefix terminated before its end at tick {int(info.get('step_count', 0))}"
            )
    return (
        env.save_state(),
        _obs5(obs),
        float(min(obs[8:16])) if len(obs) >= 16 else 1000.0,
        _pickup_bits(list(env.get_pickup_states())),
    )


def _expand_action(
    env: SpaceAceDirectEnv,
    state,
    action_idx: int,
    primitive_ticks: int,
    target_pickup: int,
) -> tuple[bool, bool, int, tuple[float, float, float, float, float], float, int, object, list[int]]:
    """Return (valid, reached, ticks, obs, min_wall, pickup_bits, state, actions)."""
    env.load_state(state)
    actions: list[int] = []
    obs_tuple = _obs5(env.get_observation())
    min_wall = 1000.0
    pickup_bits = _pickup_bits(list(env.get_pickup_states()))
    for _ in range(primitive_ticks):
        obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
        actions.append(action_idx)
        obs_tuple = _obs5(obs)
        min_wall = float(min(obs[8:16])) if len(obs) >= 16 else 1000.0
        pickup_states = list(env.get_pickup_states())
        pickup_bits = _pickup_bits(pickup_states)
        if pickup_states[target_pickup]:
            return True, True, len(actions), obs_tuple, min_wall, pickup_bits, env.save_state(), actions
        if info.get("ship_exploded") or terminated or truncated:
            return False, False, len(actions), obs_tuple, min_wall, pickup_bits, env.save_state(), actions
    return True, False, len(actions), obs_tuple, min_wall, pickup_bits, env.save_state(), actions


def _route_to_pickup(
    pf: "spaceace_rl.PyPathfinder",
    obs: tuple[float, float, float, float, float],
    target_pickup: int,
) -> Optional[tuple[float, float, float]]:
    try:
        dist, dir_x, dir_y = pf.get_distance_to_specific_pickup(obs[0], obs[1], target_pickup)
    except Exception:
        return None
    return float(dist), float(dir_x), float(dir_y)


def search_segment(
    *,
    level: int,
    prefix_actions: list[int],
    target_pickup: int,
    max_steps: int,
    primitive_ticks: int,
    max_depth: int,
    beam_width: int,
    keep_per_bucket: int,
    pos_q: float,
    vel_q: float,
    rot_q: float,
    tick_cost: float,
    target_ticks: Optional[int],
) -> Optional[SegmentResult]:
    pf = spaceace_rl.PyPathfinder(level, "grid")
    pickup_coords = [(float(x), float(y)) for x, y in pf.get_pickup_coords()]
    if not 0 <= target_pickup < len(pickup_coords):
        raise ValueError(f"target pickup {target_pickup} outside 0..{len(pickup_coords) - 1}")
    target = pickup_coords[target_pickup]

    start_state, start_obs, start_min_wall, start_bits = _load_prefix_state(
        level, prefix_actions, max_steps=max_steps
    )
    if (start_bits >> target_pickup) & 1:
        return SegmentResult(0, [], 0.0, start_obs)

    start_score = _score_to_pickup(
        start_obs,
        target,
        route=_route_to_pickup(pf, start_obs, target_pickup),
        tick_cost=tick_cost,
        ticks=0,
        min_wall=start_min_wall,
    )
    beam = [Node(start_score, 0, start_state, start_obs, start_min_wall, start_bits, ())]
    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    best: Optional[SegmentResult] = None
    t0 = time.time()

    print(
        f"start obs={tuple(round(v, 2) for v in start_obs)} "
        f"target=P{target_pickup}@({target[0]:.1f},{target[1]:.1f})"
    )

    for depth in range(1, max_depth + 1):
        candidates: list[Node] = []
        bucket_counts: dict[tuple[int, int, int, int, int, int], int] = {}
        expanded = 0
        reached_this_depth = 0

        for node in beam:
            for action_idx in range(len(ALL_ACTIONS)):
                valid, reached, dt, obs, min_wall, bits, child_state, child_actions = _expand_action(
                    env,
                    node.state,
                    action_idx,
                    primitive_ticks,
                    target_pickup,
                )
                expanded += 1
                if not valid:
                    continue

                total_ticks = node.ticks + dt
                actions = node.actions + tuple(child_actions)
                if reached:
                    reached_this_depth += 1
                    result = SegmentResult(
                        ticks=total_ticks,
                        actions=list(actions),
                        score=0.0,
                        final_obs=obs,
                    )
                    if best is None or result.ticks < best.ticks:
                        best = result
                        names = " ".join(ACTION_NAMES[a] for a in result.actions[:10])
                        print(
                            f"  reached P{target_pickup}: {result.ticks}t "
                            f"depth={depth} first=[{names}]"
                        )
                    continue

                if target_ticks is not None and total_ticks >= target_ticks:
                    continue
                if best is not None and total_ticks >= best.ticks:
                    continue

                key = _quantize(
                    obs,
                    bits,
                    pos_q=pos_q,
                    vel_q=vel_q,
                    rot_q=rot_q,
                )
                count = bucket_counts.get(key, 0)
                if count >= keep_per_bucket:
                    continue
                bucket_counts[key] = count + 1

                score = _score_to_pickup(
                    obs,
                    target,
                    route=_route_to_pickup(pf, obs, target_pickup),
                    tick_cost=tick_cost,
                    ticks=total_ticks,
                    min_wall=min_wall,
                )
                candidates.append(Node(score, total_ticks, child_state, obs, min_wall, bits, actions))

        if candidates:
            beam = heapq.nsmallest(beam_width, candidates, key=lambda n: n.score)
        else:
            beam = []

        if depth % 10 == 0 or reached_this_depth or not beam:
            best_text = f"{best.ticks}t" if best is not None else "-"
            beam_score = f"{beam[0].score:.1f}" if beam else "-"
            print(
                f"depth={depth} expanded={expanded} live={len(beam)} "
                f"reached={reached_this_depth} best={best_text} "
                f"best_score={beam_score} elapsed={time.time() - t0:.1f}s"
            )

        if not beam:
            break
        if target_ticks is not None and best is not None and best.ticks <= target_ticks:
            break

    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--prefix-json", type=Path, required=True)
    parser.add_argument("--target-pickup", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--primitive-ticks", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=130)
    parser.add_argument("--beam-width", type=int, default=2500)
    parser.add_argument("--keep-per-bucket", type=int, default=2)
    parser.add_argument("--pos-q", type=float, default=18.0)
    parser.add_argument("--vel-q", type=float, default=35.0)
    parser.add_argument("--rot-q-deg", type=float, default=10.0)
    parser.add_argument("--tick-cost", type=float, default=0.15)
    parser.add_argument("--target-ticks", type=int, default=None)
    parser.add_argument(
        "--dump-segment-json",
        type=Path,
        default=None,
        help="Write only the segment actions from prefix end to target pickup.",
    )
    parser.add_argument(
        "--dump-prefix-json",
        type=Path,
        default=None,
        help="Write prefix+segment actions through the target pickup.",
    )
    args = parser.parse_args()

    trace_level, prefix_actions = load_action_file(args.prefix_json)
    if trace_level is not None and trace_level != args.level:
        raise ValueError(f"{args.prefix_json} declares level {trace_level}, expected {args.level}")

    result = search_segment(
        level=args.level,
        prefix_actions=prefix_actions,
        target_pickup=args.target_pickup,
        max_steps=args.max_steps,
        primitive_ticks=args.primitive_ticks,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        keep_per_bucket=args.keep_per_bucket,
        pos_q=args.pos_q,
        vel_q=args.vel_q,
        rot_q=math.radians(args.rot_q_deg),
        tick_cost=args.tick_cost,
        target_ticks=args.target_ticks,
    )
    if result is None:
        print("no segment found")
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
        "prefix_ticks": len(prefix_actions),
        "total_ticks": len(prefix_actions) + result.ticks,
        "final_obs": [round(v, 3) for v in result.final_obs],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
