"""Bridge a faster TAS prefix back onto a proven incumbent suffix.

Use case:
  1. A segment-local exact search finds an earlier pickup arrival.
  2. The old suffix no longer works because the arrival state changed.
  3. Search for a short exact-control bridge from the new state to a later
     state on the incumbent trace, then append the incumbent suffix and replay
     the whole candidate.

The beam score is only a search-ordering device. A bridge is accepted only when
the complete stitched action trace replays successfully in the exact simulator
and finishes in fewer ticks than the incumbent/target.

Example:
    uv run python scripts/tas_rendezvous.py --level 7 \
      --prefix-json /tmp/spaceace_l7_p2_local.json \
      --incumbent-json /tmp/spaceace_l7_local_full.json \
      --join-start 430 --join-end 500 --beat-ticks 1356 \
      --dump-json /tmp/spaceace_l7_bridge.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spaceace.core.env import SpaceAceDirectEnv  # noqa: E402
from spaceace.ghost_actions import dump_action_file, load_action_file  # noqa: E402
from spaceace.strategies.actions import ACTION_NAMES, ALL_ACTIONS  # noqa: E402
from scripts.tas_local_search import ReplayHarness  # noqa: E402


@dataclass(frozen=True)
class TracePoint:
    tick: int
    obs: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class BeamNode:
    score: float
    state: object
    actions: tuple[int, ...]


@dataclass(frozen=True)
class BridgeResult:
    ticks: int
    join_tick: int
    bridge_ticks: int
    actions: list[int]
    score: float


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _obs5(obs) -> tuple[float, float, float, float, float]:
    return (
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        float(obs[4]),
    )


def _state_score(
    obs,
    target: TracePoint,
    *,
    vel_weight: float,
    rot_weight: float,
) -> float:
    x, y, vx, vy, rot = _obs5(obs)
    tx, ty, tvx, tvy, trot = target.obs
    pos = math.hypot(x - tx, y - ty)
    vel = math.hypot(vx - tvx, vy - tvy)
    rot_err = abs(_wrap_to_pi(rot - trot))
    return pos + vel_weight * vel + rot_weight * rot_err


def replay_trace_points(
    level: int,
    actions: list[int],
    *,
    max_steps: int,
) -> tuple[list[TracePoint], int, bool]:
    """Replay a trace and return observation points after each executed tick."""
    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    env.reset()
    points = [TracePoint(0, _obs5(env.get_observation()))]
    completed = False
    ticks = 0
    for action_idx in actions[:max_steps]:
        obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
        ticks += 1
        points.append(TracePoint(ticks, _obs5(obs)))
        if info.get("level_completed"):
            completed = True
            break
        if terminated or truncated:
            break
    return points, ticks, completed


def replay_prefix_state(
    level: int,
    prefix_actions: list[int],
    *,
    max_steps: int,
) -> tuple[object, tuple[float, float, float, float, float]]:
    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    env.reset()
    obs = env.get_observation()
    for action_idx in prefix_actions[:max_steps]:
        obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
        if info.get("ship_exploded") or terminated or truncated:
            raise ValueError(
                f"prefix terminated before its end at tick {int(info.get('step_count', 0))}"
            )
    return env.save_state(), _obs5(obs)


def search_join(
    *,
    level: int,
    prefix_actions: list[int],
    incumbent_actions: list[int],
    incumbent_points: list[TracePoint],
    incumbent_ticks: int,
    prefix_state: object,
    prefix_obs: tuple[float, float, float, float, float],
    join_tick: int,
    max_depth: int,
    beam_width: int,
    validate_top: int,
    validate_every: int,
    beat_ticks: int,
    max_steps: int,
    vel_weight: float,
    rot_weight: float,
    verbose: bool,
) -> Optional[BridgeResult]:
    if not 0 <= join_tick <= incumbent_ticks:
        raise ValueError(f"join tick {join_tick} outside incumbent trace 0..{incumbent_ticks}")

    target = incumbent_points[join_tick]
    start_score = _state_score(prefix_obs, target, vel_weight=vel_weight, rot_weight=rot_weight)
    beam = [BeamNode(start_score, prefix_state, ())]
    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    validator = ReplayHarness(level=level, max_steps=max_steps)
    best: Optional[BridgeResult] = None
    t0 = time.time()
    tested = 0

    # No bridge depth can help if even zero-depth stitching cannot beat target.
    if len(prefix_actions) + incumbent_ticks - join_tick >= beat_ticks:
        max_improving_depth = beat_ticks - len(prefix_actions) - (incumbent_ticks - join_tick) - 1
    else:
        max_improving_depth = max_depth
    max_improving_depth = min(max_depth, max_improving_depth)
    if max_improving_depth < 0:
        return None

    for depth in range(1, max_depth + 1):
        expanded: list[BeamNode] = []
        for node in beam:
            for action_idx in range(len(ALL_ACTIONS)):
                env.load_state(node.state)
                obs, _reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
                if info.get("ship_exploded") or terminated or truncated:
                    continue
                score = _state_score(
                    obs,
                    target,
                    vel_weight=vel_weight,
                    rot_weight=rot_weight,
                )
                expanded.append(
                    BeamNode(score, env.save_state(), node.actions + (action_idx,))
                )
        if not expanded:
            if verbose:
                print(f"  join {join_tick}: beam empty at depth {depth}")
            break

        expanded.sort(key=lambda n: n.score)
        beam = expanded[:beam_width]

        can_beat = len(prefix_actions) + depth + incumbent_ticks - join_tick < beat_ticks
        should_validate = (
            can_beat
            and depth <= max_improving_depth
            and (depth % validate_every == 0 or depth == max_improving_depth)
        )
        if should_validate:
            for node in beam[:validate_top]:
                stitched = (
                    prefix_actions
                    + list(node.actions)
                    + incumbent_actions[join_tick:incumbent_ticks]
                )
                result = validator.replay(stitched)
                tested += 1
                if not result.completed or result.ticks >= beat_ticks:
                    continue
                candidate = BridgeResult(
                    ticks=result.ticks,
                    join_tick=join_tick,
                    bridge_ticks=depth,
                    actions=stitched[: result.ticks],
                    score=node.score,
                )
                if best is None or candidate.ticks < best.ticks:
                    best = candidate
                    if verbose:
                        names = " ".join(ACTION_NAMES[a] for a in node.actions[:8])
                        print(
                            f"  join {join_tick}: complete {candidate.ticks}t "
                            f"bridge={depth} score={node.score:.1f} first=[{names}]"
                        )

        if verbose and depth % 10 == 0:
            print(
                f"  join {join_tick}: depth={depth} best_score={beam[0].score:.1f} "
                f"tested={tested} elapsed={time.time() - t0:.1f}s"
            )

        if depth >= max_improving_depth and not can_beat:
            break

    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--prefix-json", type=Path, required=True)
    parser.add_argument("--incumbent-json", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--join-start", type=int, required=True)
    parser.add_argument("--join-end", type=int, required=True)
    parser.add_argument("--join-step", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=90)
    parser.add_argument("--beam-width", type=int, default=600)
    parser.add_argument("--validate-top", type=int, default=80)
    parser.add_argument("--validate-every", type=int, default=2)
    parser.add_argument("--beat-ticks", type=int, default=None)
    parser.add_argument("--vel-weight", type=float, default=0.25)
    parser.add_argument("--rot-weight", type=float, default=30.0)
    parser.add_argument("--dump-json", type=Path, default=None)
    args = parser.parse_args()

    prefix_level, prefix_actions = load_action_file(args.prefix_json)
    incumbent_level, incumbent_actions = load_action_file(args.incumbent_json)
    for path, declared in ((args.prefix_json, prefix_level), (args.incumbent_json, incumbent_level)):
        if declared is not None and declared != args.level:
            raise ValueError(f"{path} declares level {declared}, expected {args.level}")

    incumbent_points, incumbent_ticks, incumbent_completed = replay_trace_points(
        args.level,
        incumbent_actions,
        max_steps=args.max_steps,
    )
    if not incumbent_completed:
        raise ValueError(f"incumbent {args.incumbent_json} does not complete")

    beat_ticks = int(args.beat_ticks or incumbent_ticks)
    prefix_state, prefix_obs = replay_prefix_state(
        args.level,
        prefix_actions,
        max_steps=args.max_steps,
    )

    print(
        f"incumbent={incumbent_ticks}t beat<{beat_ticks}t "
        f"prefix={len(prefix_actions)}t joins={args.join_start}..{args.join_end}"
    )

    best: Optional[BridgeResult] = None
    for join_tick in range(args.join_start, args.join_end + 1, args.join_step):
        max_depth = min(args.max_depth, max(1, join_tick - len(prefix_actions) + 30))
        print(f"\njoin {join_tick}: max_depth={max_depth}")
        result = search_join(
            level=args.level,
            prefix_actions=prefix_actions,
            incumbent_actions=incumbent_actions,
            incumbent_points=incumbent_points,
            incumbent_ticks=incumbent_ticks,
            prefix_state=prefix_state,
            prefix_obs=prefix_obs,
            join_tick=join_tick,
            max_depth=max_depth,
            beam_width=args.beam_width,
            validate_top=args.validate_top,
            validate_every=args.validate_every,
            beat_ticks=beat_ticks if best is None else min(beat_ticks, best.ticks),
            max_steps=args.max_steps,
            vel_weight=args.vel_weight,
            rot_weight=args.rot_weight,
            verbose=True,
        )
        if result is not None and (best is None or result.ticks < best.ticks):
            best = result
            print(
                f"new best: {best.ticks}t ({best.ticks / 60.0:.2f}s) "
                f"join={best.join_tick} bridge={best.bridge_ticks}"
            )
            if args.dump_json is not None:
                dump_action_file(args.dump_json, args.level, best.actions, best.ticks)
                print(f"checkpoint -> {args.dump_json}")

    if best is None:
        print("no validated bridge found")
        return 1

    if args.dump_json is not None:
        dump_action_file(args.dump_json, args.level, best.actions, best.ticks)
        print(f"dumped best bridge -> {args.dump_json}")

    print(json.dumps({
        "level": args.level,
        "ticks": best.ticks,
        "seconds": best.ticks / 60.0,
        "join_tick": best.join_tick,
        "bridge_ticks": best.bridge_ticks,
        "score": best.score,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
