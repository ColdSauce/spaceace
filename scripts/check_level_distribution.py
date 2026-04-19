"""Audit the curriculum for distribution biases.

Scans a range of levels, aggregates pickup positions, ship start positions,
wall orientations, and difficulty stats, then reports where the curriculum
is lopsided. Saves heatmaps and histograms so you can eyeball it.

Usage:
    # Default: all curriculum levels (3000-3164 + 5000-5824)
    uv run python scripts/check_level_distribution.py

    # Custom ranges (comma-separated, supports A-B)
    uv run python scripts/check_level_distribution.py --levels 3000-3164,5000-5024

    # Base levels only
    uv run python scripts/check_level_distribution.py --levels 3000-3164
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import spaceace_rl


def parse_levels(spec: str) -> list[int]:
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def collect(levels: list[int]) -> dict:
    data = {
        "levels_ok": [],
        "levels_failed": [],
        "pickup_norm": [],      # (nx, ny) in [0,1]^2
        "pickup_abs": [],       # (x, y) in absolute coords
        "ship_norm": [],        # (nx, ny) in [0,1]^2
        "wall_midpoints_norm": [],
        "wall_angles_deg": [],  # 0 = horizontal, 90 = vertical
        "wall_lengths": [],
        "pickups_per_level": [],
        "walls_per_level": [],
        "map_sizes": [],
        "start_to_nearest_dist": [],
        "start_to_farthest_dist": [],
        "tsp_total_dist": [],
        "any_unreachable": [],
    }

    for lvl in levels:
        try:
            g = spaceace_rl.PyGameInstance(lvl, 3000)
            g.reset()
            geo = g.get_map_geometry()
            obs = g.get_observation()
            b = geo["bounds"]
            w = b["max_x"] - b["min_x"]
            h = b["max_y"] - b["min_y"]
            data["map_sizes"].append((w, h))

            ship = (float(obs[0]), float(obs[1]))
            data["ship_norm"].append(((ship[0] - b["min_x"]) / w,
                                     (ship[1] - b["min_y"]) / h))

            pickups = geo["pickup_positions"]
            data["pickups_per_level"].append(len(pickups))
            for p in pickups:
                data["pickup_abs"].append((p[0], p[1]))
                data["pickup_norm"].append(((p[0] - b["min_x"]) / w,
                                            (p[1] - b["min_y"]) / h))

            walls = geo["map_lines"]
            data["walls_per_level"].append(len(walls))
            for wseg in walls:
                mx = (wseg[0] + wseg[2]) / 2
                my = (wseg[1] + wseg[3]) / 2
                data["wall_midpoints_norm"].append(((mx - b["min_x"]) / w,
                                                    (my - b["min_y"]) / h))
                dx = wseg[2] - wseg[0]
                dy = wseg[3] - wseg[1]
                ang = math.degrees(math.atan2(abs(dy), abs(dx)))  # 0..90
                data["wall_angles_deg"].append(ang)
                data["wall_lengths"].append(math.hypot(dx, dy))

            # Pathfinder stats
            try:
                pf = spaceace_rl.PyPathfinder(lvl, "grid")
                reach_ok, dists = pf.validate_reachability(ship[0], ship[1])
                data["any_unreachable"].append(0 if reach_ok else 1)
                if dists:
                    data["start_to_nearest_dist"].append(min(dists))
                    data["start_to_farthest_dist"].append(max(dists))
                tsp = pf.get_tsp_order(ship[0], ship[1], [False] * len(pickups))
                # crude TSP cost: sum of sequential path distances
                prev = ship
                total = 0.0
                for idx in tsp:
                    d = pf.get_distance_to_specific_pickup(prev[0], prev[1], idx)
                    total += d[0]
                    prev = pickups[idx][:2]
                data["tsp_total_dist"].append(total)
            except Exception:
                data["any_unreachable"].append(1)

            data["levels_ok"].append(lvl)
        except Exception as e:
            data["levels_failed"].append((lvl, str(e)))

    return data


def _quadrant_coverage(norm_points: list[tuple[float, float]]) -> dict:
    """Percent of points in each of 16 equally-sized grid cells (4x4)."""
    if not norm_points:
        return {}
    arr = np.asarray(norm_points)
    H, xedges, yedges = np.histogram2d(arr[:, 0], arr[:, 1], bins=4,
                                        range=[[0, 1], [0, 1]])
    H = H / H.sum()
    return {
        f"cell_{i}_{j}": float(H[i, j])
        for i in range(4) for j in range(4)
    }


def _entropy_2d(norm_points: list[tuple[float, float]], bins: int = 8) -> float:
    """Shannon entropy in bits of the 2D distribution. Uniform on 8x8 grid = 6 bits."""
    if not norm_points:
        return 0.0
    arr = np.asarray(norm_points)
    H, _, _ = np.histogram2d(arr[:, 0], arr[:, 1], bins=bins, range=[[0, 1], [0, 1]])
    total = H.sum()
    if total == 0:
        return 0.0
    p = H.ravel() / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def plot_heatmap(norm_points, title, out_path, ship_overlay=None):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#000")
    if norm_points:
        arr = np.asarray(norm_points)
        h = ax.hexbin(arr[:, 0], arr[:, 1], gridsize=30, cmap="magma",
                      extent=[0, 1, 0, 1])
        plt.colorbar(h, ax=ax, label="count")
    if ship_overlay:
        arr2 = np.asarray(ship_overlay)
        ax.scatter(arr2[:, 0], arr2[:, 1], s=6, c="#00ffff",
                   alpha=0.4, edgecolors="none", label="ship starts")
        ax.legend(loc="upper right", facecolor="#2c2c2e",
                  edgecolor="none", labelcolor="white")
    ax.set_title(title, color="white", fontsize=11)
    ax.set_xlabel("norm x (0=left, 1=right)", color="white")
    ax.set_ylabel("norm y (0=top, 1=bottom)", color="white")
    ax.invert_yaxis()  # match game screen-space
    ax.set_facecolor("#1c1c1e")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#000")
    plt.close(fig)


def plot_histograms(data, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#000")

    def _style(ax, title):
        ax.set_facecolor("#1c1c1e")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")
        ax.set_title(title, color="white")

    axes[0, 0].hist(data["pickups_per_level"], bins=range(1, 20), color="#ff9500")
    _style(axes[0, 0], "pickups per level")

    axes[0, 1].hist(data["walls_per_level"], bins=30, color="#34c759")
    _style(axes[0, 1], "wall segments per level")

    axes[0, 2].hist(data["wall_angles_deg"], bins=np.linspace(0, 90, 19),
                    color="#007aff")
    _style(axes[0, 2], "wall angle (0°=horizontal, 90°=vertical)")

    if data["start_to_nearest_dist"]:
        axes[1, 0].hist(data["start_to_nearest_dist"], bins=30, color="#ffcc00")
        _style(axes[1, 0], "start → nearest pickup distance")

    if data["start_to_farthest_dist"]:
        axes[1, 1].hist(data["start_to_farthest_dist"], bins=30, color="#ff3b30")
        _style(axes[1, 1], "start → farthest pickup distance")

    if data["tsp_total_dist"]:
        axes[1, 2].hist(data["tsp_total_dist"], bins=30, color="#af52de")
        _style(axes[1, 2], "TSP total pickup tour distance")

    fig.patch.set_facecolor("#000")
    fig.suptitle("Curriculum difficulty distributions", color="white", fontsize=13)
    out = out_dir / "01_histograms.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#000")
    plt.close(fig)
    print(f"  wrote {out}")


def report(data, out_dir, bias_threshold=0.02):
    """Print stats + flag biased regions."""
    n_ok = len(data["levels_ok"])
    n_fail = len(data["levels_failed"])
    print()
    print(f"Levels audited: {n_ok} OK, {n_fail} failed")
    if data["levels_failed"]:
        print(f"  first 5 failures: {data['levels_failed'][:5]}")
    print(f"  unreachable-pickup levels: {sum(data['any_unreachable'])}")

    # Pickup distribution entropy (uniform 8x8 = 6.0 bits)
    ent_pickup = _entropy_2d(data["pickup_norm"], bins=8)
    ent_ship = _entropy_2d(data["ship_norm"], bins=8)
    ent_wall = _entropy_2d(data["wall_midpoints_norm"], bins=8)
    print()
    print("Spatial entropy (8x8 grid, max uniform = 6.00 bits):")
    print(f"  pickups       : {ent_pickup:.2f}")
    print(f"  ship starts   : {ent_ship:.2f}")
    print(f"  wall midpoints: {ent_wall:.2f}")

    # Quadrant coverage (4x4 grid)
    cov = _quadrant_coverage(data["pickup_norm"])
    uniform_frac = 1.0 / 16
    print()
    print(f"Pickup distribution across 4x4 grid (uniform = {uniform_frac:.2%} per cell):")
    low = [(k, v) for k, v in cov.items() if v < bias_threshold]
    high = [(k, v) for k, v in sorted(cov.items(), key=lambda kv: -kv[1])[:3]]
    print(f"  top cells      : {[(k, f'{v:.1%}') for k, v in high]}")
    print(f"  under-represented cells (<{bias_threshold:.0%}): "
          f"{[(k, f'{v:.1%}') for k, v in low]}")

    # Half-map bias (top vs bottom, left vs right)
    pk = np.asarray(data["pickup_norm"]) if data["pickup_norm"] else None
    if pk is not None and len(pk):
        top = float(np.mean(pk[:, 1] < 0.5))
        bot = float(np.mean(pk[:, 1] >= 0.5))
        left = float(np.mean(pk[:, 0] < 0.5))
        right = float(np.mean(pk[:, 0] >= 0.5))
        print()
        print("Pickup half-map bias:")
        print(f"  top {top:.1%}  vs  bottom {bot:.1%}   (50/50 is uniform)")
        print(f"  left {left:.1%} vs  right {right:.1%}")
        if abs(top - 0.5) > 0.1:
            print(f"  ⚠ top/bottom imbalance of {abs(top - 0.5) * 100:.1f} points")
        if abs(left - 0.5) > 0.1:
            print(f"  ⚠ left/right imbalance of {abs(left - 0.5) * 100:.1f} points")

    # Wall orientation balance
    if data["wall_angles_deg"]:
        horiz = float(np.mean(np.array(data["wall_angles_deg"]) < 22.5))
        vert = float(np.mean(np.array(data["wall_angles_deg"]) > 67.5))
        diag = 1.0 - horiz - vert
        print()
        print("Wall orientation breakdown:")
        print(f"  horizontal (<22.5°) : {horiz:.1%}")
        print(f"  diagonal (22.5-67.5°): {diag:.1%}")
        print(f"  vertical (>67.5°)    : {vert:.1%}")

    # Difficulty curve: pickups per level as curriculum index grows
    if data["levels_ok"] and data["pickups_per_level"]:
        lvls = np.asarray(data["levels_ok"])
        pks = np.asarray(data["pickups_per_level"])
        # Correlation: does difficulty (pickup count) grow with level number?
        if len(lvls) > 1:
            corr = float(np.corrcoef(lvls, pks)[0, 1])
            print()
            print(f"Correlation(level_id, pickups_per_level) = {corr:+.2f}")
            if abs(corr) < 0.1:
                print("  (no monotonic difficulty progression)")

    # Save JSON summary
    import json
    summary = {
        "n_levels_ok": n_ok,
        "n_levels_failed": n_fail,
        "pickup_entropy_bits": ent_pickup,
        "ship_entropy_bits": ent_ship,
        "wall_entropy_bits": ent_wall,
        "pickup_quadrant_coverage": cov,
        "pickups_per_level": {
            "mean": float(np.mean(data["pickups_per_level"])) if data["pickups_per_level"] else 0,
            "min": int(min(data["pickups_per_level"])) if data["pickups_per_level"] else 0,
            "max": int(max(data["pickups_per_level"])) if data["pickups_per_level"] else 0,
        },
        "unreachable_levels": int(sum(data["any_unreachable"])),
    }
    out = out_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", default="3000-3164,5000-5824",
                   help="Comma-separated level ids or ranges")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--bias-threshold", type=float, default=0.02,
                   help="Flag cells with <N fraction of pickups (default 0.02)")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(__file__).resolve().parent.parent / "viz" / "distribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    levels = parse_levels(args.levels)
    print(f"Auditing {len(levels)} levels -> {out_dir}/")
    data = collect(levels)

    plot_heatmap(data["pickup_norm"],
                 f"Pickup density across {len(data['levels_ok'])} levels "
                 f"({len(data['pickup_norm'])} pickups total)",
                 out_dir / "02_pickup_heatmap.png",
                 ship_overlay=data["ship_norm"])
    print(f"  wrote {out_dir / '02_pickup_heatmap.png'}")

    plot_heatmap(data["ship_norm"],
                 f"Ship start density ({len(data['ship_norm'])} levels)",
                 out_dir / "03_ship_start_heatmap.png")
    print(f"  wrote {out_dir / '03_ship_start_heatmap.png'}")

    plot_heatmap(data["wall_midpoints_norm"],
                 f"Wall density ({len(data['wall_midpoints_norm'])} segments)",
                 out_dir / "04_wall_heatmap.png")
    print(f"  wrote {out_dir / '04_wall_heatmap.png'}")

    plot_histograms(data, out_dir)
    report(data, out_dir, bias_threshold=args.bias_threshold)


if __name__ == "__main__":
    main()
