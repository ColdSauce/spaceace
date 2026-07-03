#!/usr/bin/env python
"""Diagnose an action tape: per-leg splits, speed profile, thrust duty,
distance vs geodesic, and a speed-colored trajectory render.

This is the "diagnostic loop" tool from docs/SOLVER.md: run it on the
current sidecar after every solver change to see where ticks are spent.

Usage:
    uv run python scripts/analyze_tape.py --level 7                  # sidecar
    uv run python scripts/analyze_tape.py --level 7 --tape foo.json  # explicit
    uv run python scripts/analyze_tape.py --level 7 --out /tmp/l7.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spaceace_rl  # noqa: E402


def load_tape(level: int, tape_path: str | None) -> list[int]:
    if tape_path:
        d = json.load(open(tape_path))
        return d["actions"] if isinstance(d, dict) else d
    d = json.load(open(f"ghost_actions/L{level}_tas.json"))
    return d["actions"]


ACTIONS = [
    (0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (0, 1, 0), (0, 1, 1),
]


def replay_states(level: int, tape: list[int]):
    """Exact engine replay -> list of per-tick dicts (post-step state)."""
    game = spaceace_rl.PyGameInstance(level, len(tape) + 10)
    game.reset()
    states = []
    for i, a in enumerate(tape):
        obs, _, terminated, _, info = game.step(ACTIONS[a])
        states.append({
            "tick": i + 1,
            "x": float(obs[0]), "y": float(obs[1]),
            "vx": float(obs[2]), "vy": float(obs[3]),
            "rot": float(obs[4]),
            "action": a,
            "pickups_remaining": int(info["pickups_remaining"]),
            "completed": bool(info["level_completed"]),
            "exploded": bool(info["ship_exploded"]),
        })
        if terminated:
            break
    return states


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--tape", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="trajectory PNG path")
    ap.add_argument("--json", action="store_true", help="emit machine-readable summary")
    args = ap.parse_args()

    tape = load_tape(args.level, args.tape)
    states = replay_states(args.level, tape)
    last = states[-1]
    if not last["completed"]:
        print(f"WARNING: tape does not complete (exploded={last['exploded']})")

    game = spaceace_rl.PyGameInstance(args.level, 10)
    game.reset()
    geo = game.get_map_geometry()
    pickups = [(p[0], p[1]) for p in geo["pickup_positions"]]
    solver = spaceace_rl.PySolver(args.level)

    # Leg boundaries: ticks where pickups_remaining drops.
    legs = []
    prev_rem = len(pickups)
    leg_start = 0
    for s in states:
        if s["pickups_remaining"] < prev_rem:
            legs.append((leg_start, s["tick"]))
            leg_start = s["tick"]
            prev_rem = s["pickups_remaining"]

    speeds = [math.hypot(s["vx"], s["vy"]) for s in states]
    total = last["tick"]

    def leg_stats(a: int, b: int) -> dict:
        seg = states[a:b]
        sp = speeds[a:b]
        dist = sum(
            math.hypot(seg[i + 1]["x"] - seg[i]["x"], seg[i + 1]["y"] - seg[i]["y"])
            for i in range(len(seg) - 1)
        )
        thrust = sum(1 for s in seg if ACTIONS[s["action"]][2]) / max(len(seg), 1)
        rotate = sum(1 for s in seg if ACTIONS[s["action"]][0] or ACTIONS[s["action"]][1]) / max(len(seg), 1)
        return {
            "ticks": b - a,
            "seconds": (b - a) / 60.0,
            "dist_px": dist,
            "mean_speed": sum(sp) / max(len(sp), 1),
            "max_speed": max(sp) if sp else 0.0,
            "arrive_speed": sp[-1] if sp else 0.0,
            "thrust_duty": thrust,
            "rotate_duty": rotate,
        }

    # Which pickup was collected at each leg end (nearest at boundary tick).
    def pickup_at(tick: int) -> int:
        s = states[tick - 1]
        d = [math.hypot(s["x"] - px, s["y"] - py) for px, py in pickups]
        return d.index(min(d))

    summary = {"total_ticks": total, "total_seconds": total / 60.0, "legs": []}
    print(f"L{args.level} tape: {total} ticks = {total/60:.3f}s  "
          f"({len(tape)} actions in tape)")
    print(f"{'leg':<14}{'ticks':>6}{'sec':>8}{'dist':>8}{'mean_v':>8}"
          f"{'max_v':>8}{'arr_v':>8}{'thr%':>6}{'rot%':>6}")
    prev_label = "spawn"
    for (a, b) in legs:
        st = leg_stats(a, b)
        pk = pickup_at(b)
        label = f"{prev_label}->P{pk}"
        summary["legs"].append({"label": label, **st})
        print(f"{label:<14}{st['ticks']:>6}{st['seconds']:>8.3f}{st['dist_px']:>8.0f}"
              f"{st['mean_speed']:>8.0f}{st['max_speed']:>8.0f}{st['arrive_speed']:>8.0f}"
              f"{st['thrust_duty']*100:>6.0f}{st['rotate_duty']*100:>6.0f}")
        prev_label = f"P{pk}"

    # Slow sections: contiguous runs below 150 px/s lasting >= 30 ticks.
    slow = []
    run_start = None
    for i, v in enumerate(speeds):
        if v < 150.0:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= 30:
                slow.append((run_start, i))
            run_start = None
    if run_start is not None and len(speeds) - run_start >= 30:
        slow.append((run_start, len(speeds)))
    if slow:
        print("\nslow sections (>0.5s below 150 px/s):")
        for a, b in slow:
            s = states[(a + b) // 2]
            print(f"  ticks {a}-{b} ({(b-a)/60:.2f}s) around ({s['x']:.0f},{s['y']:.0f})")
    summary["slow_sections"] = [
        {"from": a, "to": b,
         "x": states[(a + b) // 2]["x"], "y": states[(a + b) // 2]["y"]}
        for a, b in slow
    ]

    if args.json:
        print(json.dumps(summary))

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import numpy as np

        xs = np.array([s["x"] for s in states])
        ys = np.array([s["y"] for s in states])
        vv = np.array(speeds)
        pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        fig, ax = plt.subplots(figsize=(9, 16))
        for w in geo["map_lines"]:
            ax.plot([w[0], w[2]], [w[1], w[3]], color="#555", lw=1.0)
        lc = LineCollection(segs, cmap="plasma", array=vv[:-1], linewidth=2.0)
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label="speed px/s", shrink=0.5)
        for i, (px, py) in enumerate(pickups):
            ax.scatter(px, py, s=180, c="cyan", edgecolors="black", zorder=4)
            ax.annotate(f"P{i}", (px, py), xytext=(8, 8),
                        textcoords="offset points", color="white", fontsize=11)
        # Tick marks every second along the path.
        for s in states[::60]:
            ax.annotate(f"{s['tick']//60}", (s["x"], s["y"]), fontsize=7,
                        color="#00ff88", zorder=5)
        b = geo["bounds"]
        ax.set_xlim(b["min_x"] - 20, b["max_x"] + 20)
        ax.set_ylim(b["max_y"] + 20, b["min_y"] - 20)
        ax.set_aspect("equal")
        ax.set_facecolor("#111")
        ax.set_title(f"L{args.level}: {total/60:.3f}s")
        fig.tight_layout()
        fig.savefig(args.out, dpi=90)
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
