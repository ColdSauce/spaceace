"""Per-leg time-optimal solver for SpaceAce. Aims at "perfect play" on a single level.

Approach:
  1. Lock pickup ordering via the Rust pathfinder's exact Held-Karp TSP.
  2. For each leg (ship state -> target pickup collected), run weighted A* on
     macro-action sequences using the real Rust simulator as the transition
     function. State dedupe via quantization of (x, y, vx, vy, rot, pickup_bits).
     Heuristic = pathfinder grid-distance to the target pickup divided by a
     generous max speed (lower bound on ticks remaining).
  3. Stitch per-leg action sequences, replay on a fresh env for deterministic
     frame capture, and save as a ghost in the dashboard DB.

Usage:
    uv run python scripts/perfect_solver.py --level 7
    uv run python scripts/perfect_solver.py --level 7 --macro 3 --weight 1.5
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

sys.path.insert(0, str(Path(__file__).parent.parent))

import spaceace_rl  # noqa: E402
from spaceace.core.env import SpaceAceDirectEnv  # noqa: E402
from spaceace.strategies.actions import ALL_ACTIONS  # noqa: E402

NUM_ACTIONS = len(ALL_ACTIONS)
TWO_PI = 2.0 * math.pi


@dataclass
class LegResult:
    actions: list[int]       # per-physics-tick action indices
    ticks: int
    target_pickup: int


def _quantize(x: float, y: float, vx: float, vy: float, rot: float,
              pickup_bits: int,
              pos_q: float, vel_q: float, rot_q: float) -> tuple:
    rot_mod = rot % TWO_PI
    return (
        int(round(x / pos_q)),
        int(round(y / pos_q)),
        int(round(vx / vel_q)),
        int(round(vy / vel_q)),
        int(round(rot_mod / rot_q)),
        pickup_bits,
    )


def _pickup_bits(pickup_states: list[bool]) -> int:
    b = 0
    for i, c in enumerate(pickup_states):
        if c:
            b |= (1 << i)
    return b


def _obs_pack(obs) -> tuple[float, float, float, float, float]:
    return float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4])


def _time_heuristic_ticks(d: float, vx: float, vy: float,
                          dir_x: float, dir_y: float, a_max: float) -> float:
    """Admissible lower bound on ticks to cover distance `d` along unit direction
    (dir_x, dir_y) with max |accel|=a_max, starting from velocity (vx,vy).

    1D double-integrator min-time: solve d = v0*t + 0.5*a*t^2 for t > 0.
        t = (-v0 + sqrt(v0^2 + 2*a*d)) / a
    Valid for any v0 (including negative) since positive root is first time
    we reach d. a_max must upper-bound the achievable acceleration toward
    target (thrust + gravity projection) — use 500 to be safe.
    """
    if d <= 0.0:
        return 0.0
    v0 = vx * dir_x + vy * dir_y
    disc = v0 * v0 + 2.0 * a_max * d
    if disc < 0.0:
        return 0.0
    t_sec = (-v0 + math.sqrt(disc)) / a_max
    if t_sec < 0.0:
        t_sec = 0.0
    return t_sec * 60.0


def _euclid_dir(x: float, y: float, tx: float, ty: float) -> tuple[float, float, float]:
    dx = tx - x
    dy = ty - y
    d = math.sqrt(dx * dx + dy * dy)
    if d <= 1e-9:
        return 0.0, 0.0, 0.0
    return d, dx / d, dy / d


def solve_leg(
    env: SpaceAceDirectEnv,
    pathfinder: "spaceace_rl.PyPathfinder",
    start_state,
    target_pickup_idx: int,
    target_x: float,
    target_y: float,
    *,
    macro: int,
    weight: float,
    a_max_heuristic: float,
    pos_q: float,
    vel_q: float,
    rot_q: float,
    open_cap: int,
    max_expansions: int,
    tick_budget: int,
    verbose: bool = True,
) -> Optional[LegResult]:
    """Weighted A* from start_state until target_pickup_idx is collected.

    Heuristic is the straight-line Euclidean distance double-integrator lower
    bound, NOT the wall-inflated BFS path distance. Euclidean is admissible
    (can't travel less than straight-line) and doesn't bias the search into
    wall-hugging trajectories.

    Returns None if no plan found within budgets. Macro action is repeated
    `macro` physics ticks per expansion.
    """
    env.load_state(start_state)
    start_obs = env.get_observation()
    start_pstates = list(env.get_pickup_states())
    start_pbits = _pickup_bits(start_pstates)
    target_bit = 1 << target_pickup_idx

    x0, y0, vx0, vy0, rot0 = _obs_pack(start_obs)
    d0, dx0, dy0 = _euclid_dir(x0, y0, target_x, target_y)
    h0 = _time_heuristic_ticks(d0, vx0, vy0, dx0, dy0, a_max_heuristic)
    # Sanity: confirm reachable via pathfinder (returns INF for unreachable maps).
    pf_dist, _, _ = pathfinder.get_distance_to_specific_pickup(x0, y0, target_pickup_idx)
    if pf_dist >= 1e18:
        if verbose:
            print(f"  leg target {target_pickup_idx}: pathfinder reports unreachable")
        return None

    # Priority queue entries: (f, counter, g_ticks, state_snapshot, parent_idx, action_idx)
    # `parent_idx` indexes into `trace`; reconstruction walks parents back.
    trace: list[tuple[int, int]] = []  # (parent_idx, action_idx); index 0 = root sentinel
    trace.append((-1, -1))

    counter = 0
    start_key = _quantize(x0, y0, vx0, vy0, rot0, start_pbits, pos_q, vel_q, rot_q)
    closed: dict[tuple, float] = {start_key: 0.0}  # key -> best g seen

    # Priority tuple: (f, h, counter, g, state, parent_idx). Tie-breaking by h
    # focuses expansion on goal-proximate nodes when f is tied.
    open_heap: list[tuple] = []
    heapq.heappush(open_heap, (weight * h0, h0, counter, 0, start_state, 0))
    counter += 1

    best_goal: Optional[tuple[float, int]] = None  # (g_ticks, trace_idx)
    expansions = 0
    t0 = time.time()

    while open_heap:
        if expansions >= max_expansions:
            if verbose:
                print(f"  leg target {target_pickup_idx}: hit max_expansions={max_expansions}")
            break

        f, _h_of_node, _, g, state, parent_idx = heapq.heappop(open_heap)
        expansions += 1

        # If we've already found a goal and this node can't beat it, stop.
        if best_goal is not None and f >= best_goal[0]:
            break
        if g > tick_budget:
            continue

        for action_idx in range(NUM_ACTIONS):
            env.load_state(state)
            action = ALL_ACTIONS[action_idx]
            crashed = False
            collected_target = False
            for _ in range(macro):
                obs, _r, terminated, truncated, info = env.step(action)
                if info.get("ship_exploded"):
                    crashed = True
                    break
                pstates_now = list(env.get_pickup_states())
                if pstates_now[target_pickup_idx]:
                    collected_target = True
                    break
                if terminated or truncated:
                    crashed = True  # treat as dead end unless goal reached
                    break
            if crashed:
                continue

            # Build trace entry for this child
            trace.append((parent_idx, action_idx))
            child_trace_idx = len(trace) - 1

            # Count actual ticks taken in this macro (could be <macro if goal hit early)
            # We approximate by `macro` — the replay phase uses the exact stored
            # action indices per macro, and the last macro will still collect the
            # target on the same tick thanks to determinism.
            child_g = g + macro
            child_state = env.save_state()

            if collected_target:
                if best_goal is None or child_g < best_goal[0]:
                    best_goal = (child_g, child_trace_idx)
                    if verbose:
                        print(f"  leg target {target_pickup_idx}: goal reached in {child_g} ticks "
                              f"(expansions={expansions}, elapsed={time.time()-t0:.1f}s)")
                continue

            if child_g > tick_budget:
                continue

            cx, cy, cvx, cvy, crot = _obs_pack(obs)
            child_pbits = _pickup_bits(list(env.get_pickup_states()))
            key = _quantize(cx, cy, cvx, cvy, crot, child_pbits, pos_q, vel_q, rot_q)
            prev_g = closed.get(key)
            if prev_g is not None and prev_g <= child_g:
                continue
            closed[key] = child_g

            d_c, dx_c, dy_c = _euclid_dir(cx, cy, target_x, target_y)
            h = _time_heuristic_ticks(d_c, cvx, cvy, dx_c, dy_c, a_max_heuristic)
            f_child = child_g + weight * h
            if best_goal is not None and f_child >= best_goal[0]:
                continue

            heapq.heappush(open_heap, (f_child, h, counter, child_g, child_state, child_trace_idx))
            counter += 1

        # Open-set pressure relief: if we grow unbounded, trim to best `open_cap`.
        if len(open_heap) > open_cap * 2:
            open_heap = heapq.nsmallest(open_cap, open_heap)
            heapq.heapify(open_heap)

    if best_goal is None:
        if verbose:
            print(f"  leg target {target_pickup_idx}: no plan found "
                  f"(expansions={expansions}, open={len(open_heap)}, elapsed={time.time()-t0:.1f}s)")
        return None

    # Reconstruct macro-action chain, then expand to per-tick actions
    macro_actions: list[int] = []
    idx = best_goal[1]
    while idx > 0:
        parent, a = trace[idx]
        macro_actions.append(a)
        idx = parent
    macro_actions.reverse()

    tick_actions: list[int] = []
    for a in macro_actions:
        tick_actions.extend([a] * macro)

    # Trim trailing ticks that happen after the target was collected. We do this
    # by replaying and stopping at collection.
    env.load_state(start_state)
    final: list[int] = []
    for a in tick_actions:
        obs, _r, terminated, truncated, info = env.step(ALL_ACTIONS[a])
        final.append(a)
        pstates = list(env.get_pickup_states())
        if pstates[target_pickup_idx]:
            break
        if info.get("ship_exploded") or terminated or truncated:
            # Unexpected — drop the leg as unusable.
            if verbose:
                print(f"  leg target {target_pickup_idx}: replay deviated (crash/truncate mid-leg)")
            return None

    if verbose:
        print(f"  leg target {target_pickup_idx}: {len(final)} ticks, "
              f"expansions={expansions}, elapsed={time.time()-t0:.1f}s")
    return LegResult(actions=final, ticks=len(final), target_pickup=target_pickup_idx)


def _multipickup_heuristic(
    x: float, y: float, vx: float, vy: float,
    pickup_bits: int,
    tsp_order: list[int],
    pickup_coords: list[tuple[float, float]],
    a_max: float,
    vmax_future: float,
) -> float:
    """Admissible lower bound on ticks remaining to collect all pickups.

    Sums:
      • Double-integrator time from current (pos, vel) to the first uncollected
        pickup in TSP order.
      • Euclidean-distance / vmax_future for each remaining pickup-to-pickup
        segment (a strict lower bound since straight-line <= any path and
        vmax_future upper-bounds achievable cruise speed).
    """
    # Walk TSP order, skip already-collected
    remaining: list[int] = [p for p in tsp_order if not (pickup_bits >> p) & 1]
    if not remaining:
        return 0.0
    # Current -> first remaining (uses velocity)
    first = remaining[0]
    tx, ty = pickup_coords[first]
    d, dx, dy = _euclid_dir(x, y, tx, ty)
    t = _time_heuristic_ticks(d, vx, vy, dx, dy, a_max)
    # Remaining pickup-to-pickup chain (ignores velocity; lower bound)
    for i in range(len(remaining) - 1):
        ax, ay = pickup_coords[remaining[i]]
        bx, by = pickup_coords[remaining[i + 1]]
        seg = math.hypot(bx - ax, by - ay)
        t += (seg / vmax_future) * 60.0
    return t


def solve_level_whole(
    env: SpaceAceDirectEnv,
    level: int,
    *,
    macro: int,
    weight: float,
    a_max_heuristic: float,
    vmax_future: float,
    pos_q: float,
    vel_q: float,
    rot_q: float,
    open_cap: int,
    max_expansions: int,
    tick_budget: int,
) -> Optional[list[int]]:
    """Whole-level weighted A*.

    State = (pos, vel, rot, pickup_bits). Goal = all pickup_bits set. Heuristic
    sums the current-leg double-integrator bound (velocity-aware) plus lower
    bounds on all remaining pickup-to-pickup segments. This penalizes
    high-speed arrivals whose residual velocity is incompatible with the next
    leg, so the search naturally coordinates across legs.
    """
    pf = spaceace_rl.PyPathfinder(level, "grid")
    pickup_coords = pf.get_pickup_coords()
    env.reset()
    start_state = env.save_state()
    initial_pstates = list(env.get_pickup_states())
    total_pickups = len(initial_pstates)
    all_collected_mask = (1 << total_pickups) - 1

    obs0 = env.get_observation()
    x0, y0, vx0, vy0, rot0 = _obs_pack(obs0)
    tsp_order = pf.get_tsp_order(x0, y0, initial_pstates)
    print(f"[solver] TSP pickup order: {tsp_order}")
    start_pbits = _pickup_bits(initial_pstates)

    h0 = _multipickup_heuristic(x0, y0, vx0, vy0, start_pbits,
                                 tsp_order, pickup_coords,
                                 a_max_heuristic, vmax_future)
    print(f"[solver] start heuristic: {h0:.0f} ticks = {h0/60.0:.2f}s")

    # Trace entries: (parent_idx, action_idx)
    trace: list[tuple[int, int]] = [(-1, -1)]
    counter = 0
    start_key = _quantize(x0, y0, vx0, vy0, rot0, start_pbits,
                          pos_q, vel_q, rot_q)
    # closed[key] = (best_g_seen, best_f_seen). Dedup accepts a new state if
    # either its g is smaller OR its f is smaller — the latter catches cases
    # where quantization collapses states that differ in residual velocity
    # (same bin, but one has better approach velocity for the next leg).
    closed: dict[tuple, tuple[float, float]] = {start_key: (0.0, weight * h0)}
    open_heap: list[tuple] = []
    heapq.heappush(open_heap, (weight * h0, h0, counter, 0, start_state, 0))
    counter += 1

    best_goal: Optional[tuple[float, int]] = None
    expansions = 0
    t0 = time.time()
    last_log = t0

    while open_heap:
        if expansions >= max_expansions:
            print(f"[solver] hit max_expansions={max_expansions}")
            break
        f, _h_n, _, g, state, parent_idx = heapq.heappop(open_heap)
        expansions += 1
        if best_goal is not None and f >= best_goal[0]:
            break
        if g > tick_budget:
            continue

        for action_idx in range(NUM_ACTIONS):
            env.load_state(state)
            action = ALL_ACTIONS[action_idx]
            crashed = False
            all_done = False
            for _ in range(macro):
                obs, _r, terminated, truncated, info = env.step(action)
                if info.get("ship_exploded"):
                    crashed = True
                    break
                if info.get("level_completed"):
                    all_done = True
                    break
                if terminated or truncated:
                    crashed = True
                    break
            if crashed:
                continue

            trace.append((parent_idx, action_idx))
            child_trace_idx = len(trace) - 1
            child_g = g + macro
            child_state = env.save_state()

            if all_done:
                if best_goal is None or child_g < best_goal[0]:
                    best_goal = (child_g, child_trace_idx)
                    print(f"[solver] goal! {child_g} ticks = {child_g/60.0:.2f}s "
                          f"(expansions={expansions}, elapsed={time.time()-t0:.1f}s)")
                    # Weighted A* (w>1) is w-optimal on first goal; bail early
                    # to avoid exhaustive re-exploration of the full frontier.
                    if weight > 1.001:
                        open_heap.clear()
                        break
                continue

            if child_g > tick_budget:
                continue

            cx, cy, cvx, cvy, crot = _obs_pack(obs)
            child_pbits = _pickup_bits(list(env.get_pickup_states()))
            key = _quantize(cx, cy, cvx, cvy, crot, child_pbits,
                            pos_q, vel_q, rot_q)

            h = _multipickup_heuristic(cx, cy, cvx, cvy, child_pbits,
                                        tsp_order, pickup_coords,
                                        a_max_heuristic, vmax_future)
            f_child = child_g + weight * h

            prev = closed.get(key)
            if prev is not None:
                prev_g, prev_f = prev
                # Prune only if both g and f are not better than recorded.
                if child_g >= prev_g and f_child >= prev_f:
                    continue
            closed[key] = (
                child_g if prev is None else min(prev[0], child_g),
                f_child if prev is None else min(prev[1], f_child),
            )

            if best_goal is not None and f_child >= best_goal[0]:
                continue
            heapq.heappush(open_heap, (f_child, h, counter, child_g,
                                         child_state, child_trace_idx))
            counter += 1

        if len(open_heap) > open_cap * 2:
            open_heap = heapq.nsmallest(open_cap, open_heap)
            heapq.heapify(open_heap)

        now = time.time()
        if now - last_log > 2.0:
            last_log = now
            if open_heap:
                best_f = open_heap[0][0]
                best_h = open_heap[0][1]
                best_g = open_heap[0][3]
                print(f"[solver] expansions={expansions} open={len(open_heap)} "
                      f"closed={len(closed)} best_f={best_f:.0f} "
                      f"(g={best_g} h={best_h:.0f}) elapsed={now-t0:.1f}s")

    if best_goal is None:
        print("[solver] no complete plan found")
        return None

    # Reconstruct
    macro_actions: list[int] = []
    idx = best_goal[1]
    while idx > 0:
        parent, a = trace[idx]
        macro_actions.append(a)
        idx = parent
    macro_actions.reverse()
    tick_actions: list[int] = []
    for a in macro_actions:
        tick_actions.extend([a] * macro)
    return tick_actions


def replay_and_capture(env: SpaceAceDirectEnv, actions: list[int]) -> tuple[list[dict], int, bool]:
    env.reset()
    frames: list[dict] = []
    total = 0
    completed = False
    for a_idx in actions:
        action = ALL_ACTIONS[a_idx]
        obs, _r, terminated, truncated, info = env.step(action)
        total += 1
        frames.append({
            "x": round(float(obs[0]), 1),
            "y": round(float(obs[1]), 1),
            "rotation": round(float(obs[4]), 3),
            "thrusting": int(action[2]) > 0,
            "tick": total,
        })
        if info.get("level_completed"):
            completed = True
            break
        if terminated or truncated:
            break
    return frames, total, completed


def build_ghost_frames(frames: list[dict]) -> list[dict]:
    ghost_frames: list[dict] = []
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
    return ghost_frames


def save_ghost(level: int, ghost_type: str, time_seconds: float, ghost_frames: list[dict]) -> None:
    from dashboard.db import get_db, init_db
    init_db()
    db = get_db()
    try:
        existing = db.execute(
            "SELECT time_seconds FROM ghost_replays WHERE level = ? AND ghost_type = ?",
            (level, ghost_type),
        ).fetchone()
        if existing and existing["time_seconds"] <= time_seconds:
            print(f"[save] existing {ghost_type} ghost is faster "
                  f"({existing['time_seconds']:.2f}s <= {time_seconds:.2f}s); not overwriting")
            return
        db.execute(
            """INSERT OR REPLACE INTO ghost_replays
               (level, ghost_type, steps, time_seconds, frames_json)
               VALUES (?, ?, ?, ?, ?)""",
            (level, ghost_type, len(ghost_frames), time_seconds, json.dumps(ghost_frames)),
        )
        db.commit()
        prev = f" (prev {existing['time_seconds']:.2f}s)" if existing else ""
        print(f"[save] wrote {ghost_type} level {level}: "
              f"{len(ghost_frames)} frames, {time_seconds:.2f}s{prev}")
    finally:
        db.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--macro", type=int, default=3,
                   help="physics ticks per macro-action (A* expansion granularity)")
    p.add_argument("--weight", type=float, default=1.1,
                   help="weighted-A* inflation factor on the heuristic")
    p.add_argument("--a-max", type=float, default=500.0,
                   help="max accel (px/s^2) upper-bound used in the double-integrator heuristic")
    p.add_argument("--vmax-future", type=float, default=500.0,
                   help="upper-bound cruise speed (px/s) for future-segment lower bound")
    p.add_argument("--pos-q", type=float, default=30.0)
    p.add_argument("--vel-q", type=float, default=40.0)
    p.add_argument("--rot-q-deg", type=float, default=15.0)
    p.add_argument("--open-cap", type=int, default=80000)
    p.add_argument("--max-expansions", type=int, default=1500000)
    p.add_argument("--tick-budget", type=int, default=6000)
    p.add_argument("--label", default="perfect")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--dump-json", default=None,
                   help="Optional path to dump the raw action sequence as JSON")
    args = p.parse_args()

    rot_q = math.radians(args.rot_q_deg)
    env = SpaceAceDirectEnv(level=args.level, max_steps=args.max_steps)

    t0 = time.time()
    actions = solve_level_whole(
        env, args.level,
        macro=args.macro, weight=args.weight,
        a_max_heuristic=args.a_max, vmax_future=args.vmax_future,
        pos_q=args.pos_q, vel_q=args.vel_q, rot_q=rot_q,
        open_cap=args.open_cap,
        max_expansions=args.max_expansions,
        tick_budget=args.tick_budget,
    )
    print(f"[solver] elapsed {time.time()-t0:.1f}s")
    if actions is None:
        return 1

    frames, ticks, completed = replay_and_capture(env, actions)
    if not completed:
        print(f"[solver] ERROR: stitched replay did not complete (ticks={ticks})")
        return 2
    time_seconds = ticks / 60.0
    print(f"[solver] final: {ticks} ticks = {time_seconds:.2f}s game-time, {len(actions)} actions")

    if args.dump_json:
        Path(args.dump_json).write_text(json.dumps({"level": args.level, "actions": actions}))
        print(f"[solver] dumped action sequence -> {args.dump_json}")

    if args.no_save:
        return 0

    ghost_frames = build_ghost_frames(frames)
    save_ghost(args.level, args.label, time_seconds, ghost_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
