"""Compare AI vs human ghost trajectories on a level.

Identifies per-pickup timing for each ghost, prints where the gap is, and
summarizes velocity/thrust patterns at the gap segments. Pure read-only
diagnostic — no engine simulation, no search, no DB writes.

    uv run python scripts/compare_ghosts.py --level 7
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_level_pickups(level: int) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """Return (pickup_positions, start_xy) by replicating real_map_parser.rs."""
    data = json.load(open(PROJECT_ROOT / "data" / "spaceace_levels.json"))
    flat = data[str(level)]
    vc = int(flat[0])
    verts = [(float(flat[1 + 2 * i]), float(flat[1 + 2 * i + 1])) for i in range(vc)]
    ol = 1 + vc * 2 + 1
    pl = int(flat[ol - 1])
    start_index_offset = ol + pl * 2
    start_idx = int(flat[start_index_offset])
    ql = ol + pl * 2 + 4
    rl = int(flat[ql - 1]) if ql - 1 < len(flat) else 0
    pickup_indices = [int(flat[ql + i]) for i in range(rl) if ql + i < len(flat)]
    return [verts[i] for i in pickup_indices], verts[start_idx]


def load_ghost(level: int, ghost_type: str) -> list[dict]:
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
        raise SystemExit(f"no {ghost_type} ghost for level {level}")
    return json.loads(row["frames_json"])


def find_pickup_times(
    frames: list[dict],
    pickups: list[tuple[float, float]],
    radius: float = 35.0,
) -> list[float | None]:
    """For each pickup (in original order), return the earliest frame time
    where the ship was within `radius` and that pickup hasn't been claimed yet
    by an earlier crossing. Returns None if never reached."""
    remaining = list(range(len(pickups)))
    out: list[float | None] = [None] * len(pickups)
    for f in frames:
        x, y, t = float(f["x"]), float(f["y"]), float(f["time"])
        if not remaining:
            break
        # Find which uncollected pickup this frame is closest to (within radius).
        best_idx = None
        best_d2 = radius * radius
        for i in remaining:
            px, py = pickups[i]
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best_idx = i
        if best_idx is not None:
            out[best_idx] = t
            remaining.remove(best_idx)
    return out


def velocity_summary(frames: list[dict]) -> dict:
    """Numerical velocity stats from successive frames."""
    speeds = []
    thrust_frac = 0
    for i in range(1, len(frames)):
        f0, f1 = frames[i - 1], frames[i]
        dt = float(f1["time"]) - float(f0["time"])
        if dt <= 0:
            continue
        dx = float(f1["x"]) - float(f0["x"])
        dy = float(f1["y"]) - float(f0["y"])
        speeds.append(math.hypot(dx, dy) / dt)
        if f0.get("thrusting"):
            thrust_frac += 1
    if not speeds:
        return {}
    speeds.sort()
    return {
        "mean_speed": sum(speeds) / len(speeds),
        "median_speed": speeds[len(speeds) // 2],
        "p90_speed": speeds[int(len(speeds) * 0.9)],
        "max_speed": speeds[-1],
        "thrust_fraction": thrust_frac / max(1, len(frames) - 1),
        "n_frames": len(frames),
    }


def velocity_in_range(frames: list[dict], t0: float, t1: float) -> dict:
    """Velocity stats restricted to frames in [t0, t1]."""
    sub = [f for f in frames if t0 <= float(f["time"]) <= t1]
    return velocity_summary(sub)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--baseline", default="ai", help="ghost_type for the slower run")
    ap.add_argument("--reference", default="human", help="ghost_type for the faster run")
    ap.add_argument("--radius", type=float, default=35.0,
                    help="px radius for declaring a pickup collected")
    args = ap.parse_args()

    pickups, start = parse_level_pickups(args.level)
    print(f"level {args.level}: {len(pickups)} pickups, start at ({start[0]:.0f}, {start[1]:.0f})")
    for i, p in enumerate(pickups):
        print(f"  pickup {i}: ({p[0]:.0f}, {p[1]:.0f})")
    print()

    base = load_ghost(args.level, args.baseline)
    ref = load_ghost(args.level, args.reference)
    base_total = float(base[-1]["time"])
    ref_total = float(ref[-1]["time"])
    print(f"{args.baseline:>8s}: {len(base)} frames, total {base_total:.2f}s")
    print(f"{args.reference:>8s}: {len(ref)} frames, total {ref_total:.2f}s")
    print(f"gap: {base_total - ref_total:+.2f}s "
          f"({(base_total - ref_total) * 60:.0f} ticks, "
          f"{(base_total - ref_total) / ref_total * 100:.1f}% slower)")
    print()

    base_t = find_pickup_times(base, pickups, args.radius)
    ref_t = find_pickup_times(ref, pickups, args.radius)

    # Re-rank pickups by collection order in each run for honest segment
    # comparison — the order may differ between human and AI.
    def order(times):
        return sorted(
            (i for i, t in enumerate(times) if t is not None),
            key=lambda i: times[i],
        )

    print("collection order:")
    print(f"  {args.baseline:>8s}: {order(base_t)}")
    print(f"  {args.reference:>8s}: {order(ref_t)}")
    print()

    # Per-pickup-segment timing (time elapsed from previous pickup to this one)
    print(f"{'pickup':>8s} | {args.baseline:>10s} | {args.reference:>10s} | {'gap':>8s}")
    print("-" * 50)
    base_order = order(base_t)
    ref_order = order(ref_t)
    base_segs = []
    ref_segs = []
    prev_b, prev_r = 0.0, 0.0
    for i in range(min(len(base_order), len(ref_order))):
        bi, ri = base_order[i], ref_order[i]
        bt = base_t[bi] - prev_b
        rt = ref_t[ri] - prev_r
        base_segs.append((bi, prev_b, base_t[bi], bt))
        ref_segs.append((ri, prev_r, ref_t[ri], rt))
        match = f"P{bi}" if bi == ri else f"P{bi}/P{ri}"
        print(f"  {match:>6s} | {bt:>8.2f}s  | {rt:>8.2f}s  | {bt - rt:>+6.2f}s")
        prev_b, prev_r = base_t[bi], ref_t[ri]
    print()

    # Velocity profile globally and on the worst segment
    print("global velocity:")
    bv = velocity_summary(base)
    rv = velocity_summary(ref)
    for k in ("mean_speed", "median_speed", "p90_speed", "max_speed", "thrust_fraction"):
        print(f"  {k:<18s}  {args.baseline}={bv.get(k, 0):>8.1f}  "
              f"{args.reference}={rv.get(k, 0):>8.1f}  "
              f"diff={(rv.get(k, 0) - bv.get(k, 0)):+.1f}")
    print()

    if base_segs and ref_segs:
        worst_idx = max(range(len(base_segs)),
                        key=lambda i: base_segs[i][3] - ref_segs[i][3])
        bi, b0, b1, bt = base_segs[worst_idx]
        ri, r0, r1, rt = ref_segs[worst_idx]
        print(f"worst segment (#{worst_idx}, {bt - rt:+.2f}s gap):")
        print(f"  {args.baseline}: pickup {bi}, t∈[{b0:.2f}, {b1:.2f}]s")
        print(f"  {args.reference}: pickup {ri}, t∈[{r0:.2f}, {r1:.2f}]s")
        bvs = velocity_in_range(base, b0, b1)
        rvs = velocity_in_range(ref, r0, r1)
        for k in ("mean_speed", "p90_speed", "thrust_fraction"):
            print(f"  {k:<18s}  {args.baseline}={bvs.get(k, 0):>8.1f}  "
                  f"{args.reference}={rvs.get(k, 0):>8.1f}")
        print()
        print(f"velocity time-series for worst segment (every 0.5s):")
        print(f"  {'t':>5s}  {args.baseline+' v':>10s} {args.baseline+' thr':>5s}  "
              f"{args.reference+' v':>10s} {args.reference+' thr':>5s}")

        def series(frames, t0, t1, step=0.5):
            sub = [f for f in frames if t0 <= float(f["time"]) <= t1]
            out = []
            t_target = t0
            for i in range(1, len(sub)):
                f0 = sub[i - 1]; f1 = sub[i]
                dt = float(f1["time"]) - float(f0["time"])
                if dt <= 0: continue
                v = math.hypot(float(f1["x"]) - float(f0["x"]),
                               float(f1["y"]) - float(f0["y"])) / dt
                if float(f1["time"]) >= t_target:
                    out.append((float(f1["time"]), v, bool(f0.get("thrusting"))))
                    t_target += step
            return out

        bs = series(base, b0, b1)
        rs = series(ref, r0, r1)
        max_n = max(len(bs), len(rs))
        for i in range(max_n):
            t_b, v_b, thr_b = bs[i] if i < len(bs) else (None, None, None)
            t_r, v_r, thr_r = rs[i] if i < len(rs) else (None, None, None)
            t_show = t_b if t_b is not None else t_r
            print(f"  {t_show:>5.2f}  "
                  f"{f'{v_b:>10.0f}' if v_b is not None else ' '*10} "
                  f"{('T' if thr_b else '.'):>5s}  "
                  f"{f'{v_r:>10.0f}' if v_r is not None else ' '*10} "
                  f"{('T' if thr_r else '.'):>5s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
