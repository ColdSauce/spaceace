#!/usr/bin/env python
"""Post-solve audit: where is the remaining time hiding?

Three detectors, all read-only:
  1. Kinematic slack — each leg's actual time vs a geometry-derived floor
     (corner-capped speeds + bang-bang between caps). Ranks legs by slack.
  2. Human-ghost splits — per-pickup cumulative deltas vs the human ghost
     (benchmark use only, per project rules).
  3. Physics flags — thrust duty, slow sections.

Optional: --probe-orders N runs short forced-order solves near the tape's
realized order; any probe that completes faster than the incumbent is an
automatic counterexample proving a ranking bug (a slower probe proves
nothing — probes are width-limited).

Usage:
    uv run python scripts/audit_tape.py --level 3
    uv run python scripts/audit_tape.py --level 3 --probe-orders 4
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import spaceace_rl  # noqa: E402

from analyze_tape import ACTIONS, load_tape, replay_states  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "dashboard" / "spaceace_dashboard.db"

# Physics constants (must match real_physics.rs; used for floors only).
A_THRUST = 400.0
A_LAT = 400.0
GRAVITY = 100.0
FLIP_T = math.pi / 4.363323
SHIP_HALF_W = 26.0


# --- geometry helpers --------------------------------------------------------

def point_seg_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    cx, cy = x1 + t * dx, y1 + t * dy
    return math.hypot(px - cx, py - cy)


def wall_clearance(x, y, walls):
    return min(point_seg_dist(x, y, *w) for w in walls)


def rdp(points, eps):
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return points
    x1, y1 = points[0]
    x2, y2 = points[-1]
    dmax, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        d = point_seg_dist(points[i][0], points[i][1], x1, y1, x2, y2)
        if d > dmax:
            dmax, idx = d, i
    if dmax <= eps:
        return [points[0], points[-1]]
    left = rdp(points[: idx + 1], eps)
    return left[:-1] + rdp(points[idx:], eps)


# --- kinematic floor ---------------------------------------------------------

def los_smooth(points, walls, margin):
    """Greedy shortcutting: connect each vertex to the furthest later vertex
    whose straight line keeps `margin` clearance. Removes the grid path's
    10px stair-stepping, which otherwise reads as hundreds of phantom
    corners and inflates the floor far above reality."""
    def clear(a, b):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(d / 12.0))
        for k in range(n + 1):
            f = k / n
            x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            if wall_clearance(x, y, walls) < margin:
                return False
        return True

    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not clear(points[i], points[j]):
            j = max(i + 1, j - max(1, (j - i) // 4))
        out.append(points[j])
        i = j
    return out


def leg_floor(polyline, walls, v_entry):
    """Corner-capped bang-bang time along a polyline (seconds).

    Optimistic by construction (k≈1.0 model): real validated tapes run
    ~10-25% above it, so use the RANKING of actual/floor across legs, not
    the absolute ratio.
    """
    pts = rdp(polyline, 6.0)
    pts = los_smooth(pts, walls, SHIP_HALF_W)
    if len(pts) < 2:
        return 0.0
    segs = []
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        d = math.hypot(dx, dy)
        if d > 1e-6:
            segs.append((d, dx / d, dy / d, pts[i]))
    if not segs:
        return 0.0

    # Speed caps at interior vertices from turn angle + clearance-fitted radius.
    caps = [math.inf] * (len(segs) + 1)
    dead = [0.0] * (len(segs) + 1)
    for i in range(1, len(segs)):
        _, ux0, uy0, _ = segs[i - 1]
        _, ux1, uy1, v0 = segs[i]
        cosang = max(-1.0, min(1.0, ux0 * ux1 + uy0 * uy1))
        theta = math.acos(cosang)
        if theta < 0.12:
            continue
        if theta > 2.6:  # near-reversal: flip time, crawl through the apex
            caps[i] = 150.0
            dead[i] = FLIP_T
            continue
        e = max(6.0, wall_clearance(v0[0], v0[1], walls) - SHIP_HALF_W)
        r = e / (1.0 / math.cos(theta / 2.0) - 1.0)
        caps[i] = math.sqrt(A_LAT * max(r, 6.0))

    # Forward/backward passes with direction-aware acceleration.
    def a_eff(uy, brake):
        # +y is down; climbing (uy<0) accelerates at 300, brakes at 500.
        g = GRAVITY * uy
        return max(120.0, A_THRUST - g if brake else A_THRUST + g)

    n = len(segs)
    v = [0.0] * (n + 1)
    v[0] = min(v_entry, caps[0] if caps[0] != math.inf else v_entry)
    for i in range(n):
        d, _, uy, _ = segs[i]
        vmax = math.sqrt(v[i] ** 2 + 2.0 * a_eff(uy, False) * d)
        v[i + 1] = min(vmax, caps[i + 1])
    for i in range(n - 1, -1, -1):
        d, _, uy, _ = segs[i]
        vmax = math.sqrt(v[i + 1] ** 2 + 2.0 * a_eff(uy, True) * d)
        if v[i] > vmax:
            v[i] = vmax
    t = sum(dead)
    for i in range(n):
        d = segs[i][0]
        t += d / max((v[i] + v[i + 1]) / 2.0, 30.0)
    return t


# --- audit -------------------------------------------------------------------

def collect_splits(track, pickups, radius=46.5):
    remaining = set(range(len(pickups)))
    out = []
    for s in track:
        for p in sorted(remaining):
            if math.hypot(s["x"] - pickups[p][0], s["y"] - pickups[p][1]) <= radius:
                remaining.discard(p)
                out.append((p, s["t"]))
        if not remaining:
            break
    return out


def human_track(level):
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT frames_json FROM ghost_replays WHERE level=? AND ghost_type='human'",
            (level,)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return [{"t": f["time"], "x": f["x"], "y": f["y"]} for f in json.loads(row[0])]


def audit(level: int, tape: list[int], probe_orders: int = 0, quiet: bool = False):
    game = spaceace_rl.PyGameInstance(level, len(tape) + 10)
    game.reset()
    geo = game.get_map_geometry()
    walls = [tuple(w) for w in geo["map_lines"]]
    pickups = [(p[0], p[1]) for p in geo["pickup_positions"]]
    pf = spaceace_rl.PyPathfinder(level, "grid")

    states = replay_states(level, tape)
    for s in states:
        s["t"] = s["tick"] / 60.0
    total_s = states[-1]["t"]

    # Legs from pickup collections.
    splits = collect_splits(states, pickups)
    legs = []
    prev_t, prev_i = 0.0, 0
    prev_label = "spawn"
    idx_of_t = {round(s["t"], 6): i for i, s in enumerate(states)}
    for p, t in splits:
        i = idx_of_t.get(round(t, 6), prev_i)
        seg = states[prev_i:i + 1]
        v0 = math.hypot(seg[0]["vx"], seg[0]["vy"]) if seg else 0.0
        # Floor the FLOWN line (subsampled), not the pathfinder's route:
        # slack then answers "how much faster could this exact line be
        # flown", immune to homotopy mismatches between grid path and tape.
        poly = [(s["x"], s["y"]) for s in seg[::3]] or [pickups[p]]
        if poly[-1] != (seg[-1]["x"], seg[-1]["y"]):
            poly.append((seg[-1]["x"], seg[-1]["y"]))
        floor = leg_floor(poly, walls, v0)
        duty = sum(1 for s in seg if ACTIONS[s["action"]][2]) / max(len(seg), 1)
        legs.append({
            "label": f"{prev_label}->P{p}", "pickup": p,
            "start_tick": prev_i, "actual_s": t - prev_t,
            "floor_s": floor, "ratio": (t - prev_t) / max(floor, 1e-6),
            "thrust_duty": duty,
        })
        prev_t, prev_i, prev_label = t, i, f"P{p}"

    # Brake-at-pickup detector: collection speed vs surrounding speeds.
    # Expert play flies through pickups at 200-400 px/s; a deep dip means
    # the rank (or geometry) is forcing a stop-and-go.
    idx_by_pickup = {}
    for p, t in splits:
        idx_by_pickup[p] = idx_of_t.get(round(t, 6), 0)
    brakes = []
    for p, i in idx_by_pickup.items():
        def _spd(j):
            j = max(0, min(len(states) - 1, j))
            return math.hypot(states[j]["vx"], states[j]["vy"])
        vin, vc, vout = _spd(i - 30), _spd(i), _spd(i + 30)
        dip = 1.0 - vc / max(max(vin, vout), 1.0)
        brakes.append({"pickup": p, "v_collect": vc, "dip": dip})

    report = {"level": level, "total_s": total_s, "legs": legs,
              "order": [p for p, _ in splits], "brakes": brakes}

    # Human comparison.
    hu = human_track(level)
    if hu:
        hs = collect_splits(hu, pickups)
        rows = []
        for i in range(min(len(splits), len(hs))):
            rows.append({"pos": i + 1, "ai": splits[i], "human": hs[i],
                         "cum_delta": splits[i][1] - hs[i][1]})
        report["human"] = {"total_s": hu[-1]["t"], "rows": rows}

    if not quiet:
        print(f"L{level} audit: {total_s:.3f}s, order {report['order']}")
        print(f"{'leg':<14}{'actual':>8}{'floor':>8}{'ratio':>7}{'duty%':>7}   slack rank")
        for rank, leg in enumerate(sorted(legs, key=lambda x: -(x['actual_s'] - x['floor_s']))):
            pass
        ranked = sorted(legs, key=lambda x: -(x["actual_s"] - x["floor_s"]))
        rank_of = {id(l): i + 1 for i, l in enumerate(ranked)}
        for leg in legs:
            print(f"{leg['label']:<14}{leg['actual_s']:>8.2f}{leg['floor_s']:>8.2f}"
                  f"{leg['ratio']:>7.2f}{leg['thrust_duty'] * 100:>7.0f}"
                  f"   #{rank_of[id(leg)]}"
                  + ("  <-- fattest" if rank_of[id(leg)] == 1 else ""))
        hard_brakes = [b for b in brakes if b["dip"] > 0.5 and b["v_collect"] < 120]
        if hard_brakes:
            print("brake-at-pickup: " + ", ".join(
                f"P{b['pickup']} ({b['v_collect']:.0f}px/s, dip {b['dip'] * 100:.0f}%)"
                for b in sorted(hard_brakes, key=lambda b: b["v_collect"])))
        if hu:
            worst = max(report["human"]["rows"], key=lambda r: r["cum_delta"], default=None)
            lead = [r for r in report["human"]["rows"] if r["cum_delta"] > 0.05]
            print(f"human ghost: {report['human']['total_s']:.2f}s; "
                  f"human ahead at {len(lead)}/{len(report['human']['rows'])} splits"
                  + (f", worst cum delta +{worst['cum_delta']:.2f}s at split {worst['pos']}"
                     if worst and worst["cum_delta"] > 0.05 else ""))

    # Forced-order counterexample probes.
    if probe_orders > 0:
        solver = spaceace_rl.PySolver(level)
        order = report["order"]
        cands = []
        for i in range(len(order) - 1):
            o = order[:]
            o[i], o[i + 1] = o[i + 1], o[i]
            cands.append(o)
        cands = cands[:probe_orders]
        report["probes"] = []
        for o in cands:
            t = solver.solve(width=30_000, max_ticks=len(tape) + 60, seed=1,
                             mix=1.0, proj_div=300.0, lattice=True,
                             order=[int(v) for v in o])
            res = {"order": o, "ticks": len(t) if t else None}
            report["probes"].append(res)
            if not quiet:
                verdict = ("COUNTEREXAMPLE: beats incumbent!" if t and len(t) < len(tape)
                           else "no evidence" if not t or len(t) >= len(tape) else "")
                print(f"probe order {o}: "
                      f"{'no completion' if not t else f'{len(t)} ticks'}  {verdict}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--tape", type=str, default=None)
    ap.add_argument("--probe-orders", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    tape = load_tape(args.level, args.tape)
    report = audit(args.level, tape, args.probe_orders, quiet=args.json)
    if args.json:
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
