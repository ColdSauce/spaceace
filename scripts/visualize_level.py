"""Visualize a level's geometry, pathfinder output, and agent observations.

Usage:
    uv run python scripts/visualize_level.py 5115
    uv run python scripts/visualize_level.py 5115 --out-dir /tmp/viz5115
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


# Colors
SHIP_COLOR = "#ff3b30"
PATH_COLORS = ["#ff9500", "#ffcc00", "#34c759", "#007aff"]
PICKUP_COLOR = "#00ffff"
WALL_COLOR = "#8e8e93"


def _setup_map_axes(ax, bounds, walls, pickups, ship, title):
    for w in walls:
        ax.plot([w[0], w[2]], [w[1], w[3]], color=WALL_COLOR, lw=2)
    for i, p in enumerate(pickups):
        ax.scatter(p[0], p[1], s=200, c=PICKUP_COLOR, edgecolors="black",
                   linewidths=1.5, zorder=4)
        ax.annotate(f"P{i}", (p[0], p[1]), xytext=(8, 8),
                    textcoords="offset points", fontsize=9, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))
    ax.scatter(ship[0], ship[1], s=250, c=SHIP_COLOR, marker="^",
               edgecolors="white", linewidths=1.5, zorder=5, label="ship")
    ax.set_xlim(bounds["min_x"] - 20, bounds["max_x"] + 20)
    ax.set_ylim(bounds["max_y"] + 20, bounds["min_y"] - 20)  # Y flipped (screen coords)
    ax.set_aspect("equal")
    ax.set_facecolor("#1c1c1e")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.1)


def plot_overview(level, out_dir):
    g = spaceace_rl.PyGameInstance(level, 3000)
    g.reset()
    geo = g.get_map_geometry()
    obs = g.get_observation()
    ship = (float(obs[0]), float(obs[1]))

    pf = spaceace_rl.PyPathfinder(level, "grid")
    pickups = pf.get_pickup_coords()
    tsp = pf.get_tsp_order(ship[0], ship[1], [False] * len(pickups))

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#000")
    _setup_map_axes(ax, geo["bounds"], geo["map_lines"], pickups, ship,
                    f"L{level}: map + pathfinder TSP order {tsp}")

    for order_idx, pickup_idx in enumerate(tsp):
        path = pf.get_path_to_specific_pickup(ship[0], ship[1], pickup_idx)
        if not path:
            continue
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color=PATH_COLORS[order_idx % len(PATH_COLORS)],
                lw=2.5, alpha=0.9,
                label=f"{order_idx+1}. start → P{pickup_idx} ({len(path)} pts)")

    ax.legend(loc="upper right", facecolor="#2c2c2e", edgecolor="none",
              labelcolor="white", fontsize=9)
    out = out_dir / f"{level}_01_overview.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#000")
    plt.close(fig)
    print(f"  wrote {out}")
    return ship, pickups, geo, pf


def plot_paths_individually(level, ship, pickups, geo, pf, out_dir):
    """Separate subplot per pickup so overlapping paths don't hide each other."""
    n = len(pickups)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 6), facecolor="#000")
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        _setup_map_axes(ax, geo["bounds"], geo["map_lines"], pickups, ship,
                        f"path to pickup {i} @ ({pickups[i][0]:.0f}, {pickups[i][1]:.0f})")
        try:
            path = pf.get_path_to_specific_pickup(ship[0], ship[1], i)
        except Exception:
            path = []
        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color=PATH_COLORS[i % len(PATH_COLORS)], lw=3,
                    alpha=0.95, label=f"{len(path)} pts")
            dist = pf.get_distance_to_specific_pickup(ship[0], ship[1], i)
            ax.set_title(f"P{i}: path_dist={dist[0]:.0f}, dir=({dist[1]:.2f}, {dist[2]:.2f})",
                         fontsize=10)
        else:
            ax.set_title(f"P{i}: UNREACHABLE", color="red")
    out = out_dir / f"{level}_02_paths.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#000")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_raycasts(level, ship, pickups, geo, out_dir, sample_positions=None):
    """24 wall raycasts visualized from one or more ship positions.

    Rust observation puts 8 coarse rays at indices 8..16 (45° spacing starting
    forward) and 16 fine rays at 20..36 (interleaved with 15° spacing).
    """
    g = spaceace_rl.PyGameInstance(level, 3000)
    g.reset()

    positions = sample_positions or [ship]
    fig, axes = plt.subplots(1, len(positions), figsize=(7 * len(positions), 7),
                              facecolor="#000")
    if len(positions) == 1:
        axes = [axes]

    for ax, pos in zip(axes, positions):
        # Teleport ship via snapshot + direct step-less eval: cheapest is to
        # build a fresh game and manually step toward the target. Simpler: for
        # the "start" position we just use obs as-is.
        obs = g.get_observation()
        if abs(obs[0] - pos[0]) > 0.1 or abs(obs[1] - pos[1]) > 0.1:
            # Can't trivially teleport; note and skip the re-sample
            ax.text(0.5, 0.5, f"(teleport unsupported; showing start obs)",
                    transform=ax.transAxes, color="yellow", ha="center")

        _setup_map_axes(ax, geo["bounds"], geo["map_lines"], pickups,
                        (obs[0], obs[1]),
                        f"raycasts @ ({obs[0]:.0f}, {obs[1]:.0f})  rotation={obs[4]:.2f}rad")

        # Wall distances: 8 coarse + 16 fine = 24 total at 15° spacing.
        # Coarse rays (obs[8..16]) are every 45° starting at the ship's forward.
        # Fine rays (obs[20..36]) fill in the 15° & 30° offsets between them.
        # We'll just visualize all 24 at 15° increments starting at rotation.
        rotation = float(obs[4])
        # Interleaved ordering: [c0, f0, f1, c1, f2, f3, c2, f4, f5, ...]
        coarse = [float(obs[8 + i]) for i in range(8)]
        fine = [float(obs[20 + i]) for i in range(16)]
        # Build 24 distances in angle order
        dists = []
        for c in range(8):
            dists.append(coarse[c])
            dists.append(fine[c * 2])
            dists.append(fine[c * 2 + 1])
        # That produces 24 rays — but the relative angles of fine rays to coarse
        # may vary per implementation. Use 15° uniform spacing as a reasonable
        # visualization since we don't have the exact angle table from Rust.

        for i, d in enumerate(dists):
            ang = rotation + (i * 15.0) * math.pi / 180.0
            # Clip absurd distances to bounds diagonal for visualization
            dv = min(d, 1500.0)
            end_x = obs[0] + math.cos(ang) * dv
            end_y = obs[1] - math.sin(ang) * dv  # screen-y inverted
            c = "#ff9500" if i % 3 == 0 else "#ffcc00"  # coarse highlight
            ax.plot([obs[0], end_x], [obs[1], end_y], color=c, lw=1, alpha=0.7)
            # Show distance label on a few rays
            if i % 6 == 0:
                ax.annotate(f"{d:.0f}",
                            ((obs[0] + end_x) / 2, (obs[1] + end_y) / 2),
                            color="white", fontsize=7)

    out = out_dir / f"{level}_03_raycasts.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#000")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_observation_vector(level, out_dir):
    """Use the actual PathAugmentedObs23 strategy to produce the 40-dim obs
    the PPO agent receives, and bar-plot it with labels."""
    from spaceace.strategies.observation import PathAugmentedObs23
    from spaceace.strategies.pathfinder import RustPathfinder
    from spaceace.core.gym_wrapper import SpaceAceGymWrapper

    env = SpaceAceGymWrapper(level=level, max_steps=3000)
    raw_obs, info = env.reset()
    pf = RustPathfinder(level)
    obs_builder = PathAugmentedObs23(pf, 3000)
    vec = obs_builder.reset(raw_obs, info, env)

    LABELS = [
        "vx/300", "vy/300", "sin(rot)", "cos(rot)", "pickup_d/1000",
        "wall0/1k", "wall1/1k", "wall2/1k", "wall3/1k",
        "wall4/1k", "wall5/1k", "wall6/1k", "wall7/1k",
        "pickups_rem/10", "norm_x", "norm_y",
        "pf_dist/5k", "pf_dir_x", "pf_dir_y",
        "speed", "speed_toward", "heading_align", "min_tti/2", "time_remaining",
    ] + [f"fine_ray{i}" for i in range(16)]

    assert len(LABELS) == 40, f"label count drift: {len(LABELS)}"

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#000")
    colors = ["#ff9500"] * 5 + ["#8e8e93"] * 8 + ["#00ffff"] * 3 + ["#34c759"] * 8 + ["#ffcc00"] * 16
    ax.bar(range(40), vec, color=colors)
    ax.set_xticks(range(40))
    ax.set_xticklabels(LABELS, rotation=75, ha="right", fontsize=7, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("white")
    ax.set_facecolor("#1c1c1e")
    ax.set_title(f"L{level} — PathAugmentedObs23 (40-dim) at ship start",
                 color="white", fontsize=11)
    ax.axhline(0, color="white", lw=0.5, alpha=0.3)

    # Annotate each bar with its value
    for i, v in enumerate(vec):
        ax.text(i, v + (0.02 if v >= 0 else -0.04), f"{v:.2f}",
                ha="center", color="white", fontsize=6,
                rotation=90 if abs(v) < 0.2 else 0)

    out = out_dir / f"{level}_04_obs_vector.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#000")
    plt.close(fig)
    print(f"  wrote {out}")

    # Also print summary to stdout
    print()
    print(f"  PathAugmentedObs23 @ start (40 dims):")
    for i, (label, v) in enumerate(zip(LABELS, vec)):
        print(f"    [{i:2d}] {label:<18s} = {v:+.4f}")


def plot_reachability_grid(level, ship, pickups, geo, out_dir):
    """Approximate the pathfinder's reachable-cell mask by querying paths
    densely and marking which points have a finite distance.
    """
    bounds = geo["bounds"]
    pf = spaceace_rl.PyPathfinder(level, "grid")

    # For each pickup, get its path and overlay all waypoints as reachable.
    # This is a reachability UNION, not the true BFS grid, but shows where
    # the pathfinder "knows" it can travel.
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#000")
    _setup_map_axes(ax, bounds, geo["map_lines"], pickups, ship,
                    f"L{level} — union of pathfinder waypoints (reachable zone proxy)")

    all_points_x = []
    all_points_y = []
    for i in range(len(pickups)):
        try:
            path = pf.get_path_to_specific_pickup(ship[0], ship[1], i)
        except Exception:
            continue
        for wp in path:
            all_points_x.append(wp[0])
            all_points_y.append(wp[1])

    ax.scatter(all_points_x, all_points_y, s=8, c="#00ff88", alpha=0.4,
               edgecolors="none", label="reachable (on some path)")
    ax.legend(loc="upper right", facecolor="#2c2c2e", edgecolor="none",
              labelcolor="white", fontsize=9)

    out = out_dir / f"{level}_05_reachable.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#000")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("level", type=int)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(__file__).resolve().parent.parent / "viz" / f"level_{args.level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing visualizations for level {args.level} -> {out_dir}/")

    print()
    print("[1/5] overview with TSP paths")
    ship, pickups, geo, pf = plot_overview(args.level, out_dir)
    print("[2/5] per-pickup paths")
    plot_paths_individually(args.level, ship, pickups, geo, pf, out_dir)
    print("[3/5] raycasts at ship start")
    plot_raycasts(args.level, ship, pickups, geo, out_dir)
    print("[4/5] 40-dim observation vector (PathAugmentedObs23)")
    plot_observation_vector(args.level, out_dir)
    print("[5/5] reachability union")
    plot_reachability_grid(args.level, ship, pickups, geo, out_dir)

    print()
    print(f"Done. View with:  open {out_dir}")


if __name__ == "__main__":
    main()
