"""Curriculum-aware difficulty scoring for Space Ace levels.

Classifies levels into curriculum stages that introduce one concept at a time:
  Stage 1: Open — few walls, 1 pickup (learn thrust/rotation/gravity)
  Stage 2: Open + pickups — few walls, multiple pickups (learn route planning)
  Stage 3: Light maze — moderate walls, 1-2 pickups (learn wall avoidance)
  Stage 4: Light maze + pickups — moderate walls, multiple pickups
  Stage 5: Dense maze — many walls/tight corridors, few pickups (precision nav)
  Stage 6: Dense maze + pickups — many walls, many pickups (hardest)

Within each stage, levels are sorted by a difficulty score based on route length,
detour ratio, maneuver count, and upward travel.

Usage:
    uv run python -m spaceace.tools.difficulty --all
    uv run python -m spaceace.tools.difficulty --all --output data/curriculum.json
    uv run python -m spaceace.tools.difficulty --level 6
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

from tqdm import tqdm

import spaceace_rl

LEVELS_PATH = Path(__file__).parent.parent.parent / "data" / "spaceace_levels.json"


@dataclass
class LevelInfo:
    level: int
    stage: int
    stage_name: str
    intra_stage_score: float  # difficulty within the stage, for sorting
    num_walls: int
    num_pickups: int
    bottleneck_clearance: float
    detour_ratio: float
    total_route_length: float
    maneuver_count: int
    upward_travel: float


def get_all_level_ids() -> list[int]:
    with open(LEVELS_PATH) as f:
        data = json.load(f)
    return sorted(int(k) for k in data.keys() if k.isdigit())


def analyze_level(level: int) -> dict | None:
    """Get geometric difficulty metrics from the Rust pathfinder."""
    try:
        game = spaceace_rl.PyGameInstance(level, 3000)
        geom = game.get_map_geometry()
        lines = [(l[0], l[1], l[2], l[3]) for l in geom["map_lines"]]
        pickups = geom["pickup_positions"]
        if not pickups:
            return None

        info = game.get_info()
        sp = info["ship_position"]
        sx, sy = float(sp["x"]), float(sp["y"])

        pf = spaceace_rl.PyPathfinder(level)
        raw = dict(pf.analyze_level_difficulty(sx, sy, lines))
        raw["level"] = level
        return raw
    except Exception:
        return None


# --- Stage classification ---
# Maze complexity thresholds (based on num_walls distribution: p25=10, median=21, p75=42)
OPEN_MAX_WALLS = 10
LIGHT_MAZE_MAX_WALLS = 35
# Above 35 = dense maze

# Pickup thresholds
FEW_PICKUPS_MAX = 2


def classify_stage(raw: dict) -> tuple[int, str]:
    """Classify a level into a curriculum stage based on its geometry."""
    walls = raw["num_walls"]
    pickups = raw["num_pickups"]
    few_pickups = pickups <= FEW_PICKUPS_MAX

    if walls <= OPEN_MAX_WALLS:
        if few_pickups:
            return 1, "open"
        else:
            return 2, "open+pickups"
    elif walls <= LIGHT_MAZE_MAX_WALLS:
        if few_pickups:
            return 3, "light-maze"
        else:
            return 4, "light-maze+pickups"
    else:
        if few_pickups:
            return 5, "dense-maze"
        else:
            return 6, "dense-maze+pickups"


def intra_stage_difficulty(raw: dict) -> float:
    """Compute a difficulty score within a stage for sorting.

    Combines route length, detour ratio, maneuver count, and upward travel
    into a single number. Higher = harder.
    """
    route = raw.get("total_route_length", 0)
    detour = raw.get("detour_ratio", 1.0)
    maneuvers = raw.get("maneuver_count", 0)
    upward = raw.get("upward_travel", 0)
    bottleneck = raw.get("bottleneck_clearance", 200)

    # Tightness component: how close to ship radius (36.5px)
    tightness = max(0, 1.0 - (bottleneck - 36.5) / 110)  # 0 at 146px+, 1 at 36.5px

    # Weighted combination (route length dominates within a stage)
    score = (
        route * 1.0
        + (detour - 1.0) * 500    # detour of 2.0 adds 500
        + maneuvers * 100          # each maneuver point adds 100
        + upward * 0.5             # upward travel adds half its value
        + tightness * 300          # tight corridors add up to 300
    )
    return score


def run_all(level_ids: list[int] | None = None) -> list[LevelInfo]:
    """Analyze all levels and return curriculum-ordered list."""
    if level_ids is None:
        level_ids = get_all_level_ids()

    results = []
    for lid in tqdm(level_ids, desc="Analyzing levels", unit="level"):
        raw = analyze_level(lid)
        if raw is None:
            continue
        stage, stage_name = classify_stage(raw)
        score = intra_stage_difficulty(raw)
        results.append(LevelInfo(
            level=lid,
            stage=stage,
            stage_name=stage_name,
            intra_stage_score=round(score, 1),
            num_walls=int(raw["num_walls"]),
            num_pickups=int(raw["num_pickups"]),
            bottleneck_clearance=round(raw["bottleneck_clearance"], 1),
            detour_ratio=round(raw["detour_ratio"], 3),
            total_route_length=round(raw["total_route_length"], 0),
            maneuver_count=int(raw["maneuver_count"]),
            upward_travel=round(raw["upward_travel"], 0),
        ))

    # Sort: primary by stage, secondary by intra-stage difficulty
    results.sort(key=lambda r: (r.stage, r.intra_stage_score))
    return results


def main():
    parser = argparse.ArgumentParser(description="Space Ace curriculum difficulty scorer")
    parser.add_argument("--level", type=int, help="Analyze a single level")
    parser.add_argument("--all", action="store_true", help="Analyze all levels")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    args = parser.parse_args()

    if args.level is not None:
        raw = analyze_level(args.level)
        if raw is None:
            print(f"Failed to analyze level {args.level}")
            sys.exit(1)
        stage, stage_name = classify_stage(raw)
        score = intra_stage_difficulty(raw)

        print(f"\nLevel {args.level}")
        print("=" * 40)
        print(f"Stage:       {stage} ({stage_name})")
        print(f"Intra-score: {score:.0f}")
        print(f"Walls:       {raw['num_walls']}")
        print(f"Pickups:     {raw['num_pickups']}")
        print(f"Bottleneck:  {raw['bottleneck_clearance']:.1f} px")
        print(f"Detour:      {raw['detour_ratio']:.2f}")
        print(f"Route:       {raw['total_route_length']:.0f} px")
        print(f"Maneuvers:   {raw['maneuver_count']}")
        print(f"Upward:      {raw['upward_travel']:.0f} px")

    elif args.all:
        results = run_all()

        # Print stage summary
        from collections import Counter
        stage_counts = Counter(r.stage for r in results)
        stage_names = {r.stage: r.stage_name for r in results}

        print(f"\n{'='*70}")
        print("CURRICULUM STAGES")
        print(f"{'='*70}")
        for s in sorted(stage_counts.keys()):
            name = stage_names.get(s, "?")
            count = stage_counts[s]
            levels_in_stage = [r for r in results if r.stage == s]
            easiest = levels_in_stage[0] if levels_in_stage else None
            hardest = levels_in_stage[-1] if levels_in_stage else None
            print(f"\nStage {s}: {name} ({count} levels)")
            print(f"  {'Level':>8} {'Score':>8} {'Walls':>6} {'Pick':>5} {'Bottle':>7} {'Detour':>7} {'Route':>7} {'Manvr':>6}")
            # Show first 5 and last 5
            show = levels_in_stage[:5]
            if len(levels_in_stage) > 10:
                show += [None]  # separator
                show += levels_in_stage[-5:]
            elif len(levels_in_stage) > 5:
                show = levels_in_stage
            for r in show:
                if r is None:
                    print(f"  {'...':>8}")
                    continue
                print(f"  {r.level:>8} {r.intra_stage_score:>8.0f} {r.num_walls:>6} {r.num_pickups:>5} "
                      f"{r.bottleneck_clearance:>7.1f} {r.detour_ratio:>7.2f} {r.total_route_length:>7.0f} {r.maneuver_count:>6}")

        print(f"\n{'='*70}")
        print(f"Total: {len(results)} levels across {len(stage_counts)} stages")

        # Save output
        if args.output:
            output_path = Path(args.output)
            output_data = {
                "curriculum_order": [r.level for r in results],
                "stages": {},
                "levels": {},
            }
            for s in sorted(stage_counts.keys()):
                levels_in_stage = [r for r in results if r.stage == s]
                output_data["stages"][str(s)] = {
                    "name": stage_names[s],
                    "count": len(levels_in_stage),
                    "levels": [r.level for r in levels_in_stage],
                }
            for r in results:
                output_data["levels"][str(r.level)] = asdict(r)

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nSaved to {output_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
