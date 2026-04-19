#!/usr/bin/env python3
"""Clone existing levels with additional pickups for curriculum training.

Takes a range of source levels and creates variants with more pickups placed
at safe positions inside the level walls. Uses ray-casting point-in-polygon
to ensure pickups land inside the same walled region as the spawn point.

Numbering: for source level S and extra count E (1-5):
    start_level + (S - source_start) * 5 + (E - 1)

Usage:
    uv run python -m spaceace.tools.add_pickups --source 3000-3049 --start-level 5000
"""

import argparse
import json
import math
import random
from pathlib import Path

import spaceace_rl

LEVELS_PATH = Path("data/spaceace_levels.json")
MIN_WALL_DIST = 45.0
MIN_PICKUP_DIST = 50.0
SPAWN_Y_OFFSET = -100.0


def parse_level(arr: list) -> dict:
    """Parse a flat-array level into structured data."""
    idx = 0
    n_verts = int(arr[idx]); idx += 1
    verts = [(arr[idx + i * 2], arr[idx + 1 + i * 2]) for i in range(n_verts)]
    idx += n_verts * 2
    n_lines = int(arr[idx]); idx += 1
    lines = [(int(arr[idx + i * 2]), int(arr[idx + 1 + i * 2])) for i in range(n_lines)]
    idx += n_lines * 2
    start_idx = int(arr[idx]); idx += 1
    bw = arr[idx]; idx += 1
    bh = arr[idx]; idx += 1
    n_pickups = int(arr[idx]); idx += 1
    pickups = [int(arr[idx + i]) for i in range(n_pickups)]
    idx += n_pickups
    return {
        "verts": verts, "lines": lines, "start_idx": start_idx,
        "bw": bw, "bh": bh, "pickups": pickups,
    }


def serialize_level(data: dict) -> list:
    """Serialize structured level data back to flat array."""
    arr = [len(data["verts"])]
    for x, y in data["verts"]:
        arr.extend([x, y])
    arr.append(len(data["lines"]))
    for va, vb in data["lines"]:
        arr.extend([va, vb])
    arr.append(data["start_idx"])
    arr.append(data["bw"])
    arr.append(data["bh"])
    arr.append(len(data["pickups"]))
    for p in data["pickups"]:
        arr.append(p)
    arr.append(0)
    return arr


def point_to_segment_dist(px, py, x1, y1, x2, y2):
    """Distance from point to line segment."""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-10:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x, proj_y = x1 + t * dx, y1 + t * dy
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _ray_crossings(px, py, segments):
    """Count how many wall segments a horizontal ray from (px, py) to +inf crosses."""
    crossings = 0
    for x1, y1, x2, y2 in segments:
        # Segment must straddle py vertically
        if (y1 <= py < y2) or (y2 <= py < y1):
            # X of intersection with horizontal line y=py
            t = (py - y1) / (y2 - y1)
            ix = x1 + t * (x2 - x1)
            if ix > px:
                crossings += 1
    return crossings


def _is_inside_walls(px, py, segments):
    """Point-in-polygon via ray casting. Odd crossings = inside."""
    return _ray_crossings(px, py, segments) % 2 == 1


def _build_wall_segments(data):
    """Convert vertex-index wall pairs into (x1,y1,x2,y2) tuples."""
    verts = data["verts"]
    return [(verts[a][0], verts[a][1], verts[b][0], verts[b][1]) for a, b in data["lines"]]


def add_pickups_to_level(data: dict, extra_count: int, rng: random.Random) -> dict:
    """Clone a level and add extra pickups inside the walled region."""
    verts = list(data["verts"])
    pickups = list(data["pickups"])
    segments = _build_wall_segments(data)
    bw, bh = data["bw"], data["bh"]

    existing_positions = [verts[p] for p in pickups]
    existing_positions.append(verts[data["start_idx"]])

    # Sample within the actual vertex bounding box — `bw`/`bh` do not always
    # enclose the walls (e.g. L3106 has verts offset to x=[655,1615] but
    # bw=1060, so (0,bw)×(0,bh) misses the walled region entirely).
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    for _ in range(extra_count):
        for _attempt in range(50000):
            px = rng.uniform(xmin, xmax)
            py = rng.uniform(ymin, ymax)

            # Must be inside the walled area
            if not _is_inside_walls(px, py, segments):
                continue

            # Must be far from walls
            too_close = False
            for a, b in data["lines"]:
                x1, y1 = verts[a]
                x2, y2 = verts[b]
                if point_to_segment_dist(px, py, x1, y1, x2, y2) < MIN_WALL_DIST:
                    too_close = True
                    break
            if too_close:
                continue

            # Must be far from existing pickups
            too_close = False
            for ex, ey in existing_positions:
                if math.sqrt((px - ex) ** 2 + (py - ey) ** 2) < MIN_PICKUP_DIST:
                    too_close = True
                    break
            if too_close:
                continue

            # Must be reachable from spawn via pathfinder
            trial_verts = verts + [(px, py)]
            trial_pickups = pickups + [len(verts)]
            trial_data = {
                "verts": trial_verts, "lines": data["lines"],
                "start_idx": data["start_idx"], "bw": bw, "bh": bh,
                "pickups": trial_pickups,
            }
            map_json = json.dumps(serialize_level(trial_data))
            pf = spaceace_rl.PyPathfinder.from_map_json(map_json)
            sx, sy = trial_verts[data["start_idx"]]
            all_ok, _ = pf.validate_reachability(sx, sy + SPAWN_Y_OFFSET)
            if not all_ok:
                continue

            break
        else:
            return None  # Could not place pickup — caller should skip this variant

        new_idx = len(verts)
        verts.append((px, py))
        pickups.append(new_idx)
        existing_positions.append((px, py))

    return {
        "verts": verts, "lines": data["lines"], "start_idx": data["start_idx"],
        "bw": bw, "bh": bh, "pickups": pickups,
    }


def parse_range(spec: str) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(spec)]


def main():
    parser = argparse.ArgumentParser(description="Clone levels with extra pickups")
    parser.add_argument("--source", type=str, required=True,
                        help="Source level range (e.g. 3000-3049)")
    parser.add_argument("--max-extra", type=int, default=5,
                        help="Generate variants with +1 through +N pickups (default: 5)")
    parser.add_argument("--start-level", type=int, default=5000,
                        help="Starting level number for new levels (default: 5000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_levels = parse_range(args.source)
    rng = random.Random(args.seed)

    with open(LEVELS_PATH) as f:
        all_levels = json.load(f)

    old_keys = [k for k in all_levels if not k.startswith("_") and int(k) >= args.start_level]
    for k in old_keys:
        del all_levels[k]
    if old_keys:
        print(f"Removed {len(old_keys)} old levels >= {args.start_level}")

    created = 0
    for src_idx, src_lvl in enumerate(source_levels):
        key = str(src_lvl)
        if key not in all_levels:
            continue

        data = parse_level(all_levels[key])
        base_pickups = len(data["pickups"])

        prev_data = data
        for extra in range(1, args.max_extra + 1):
            level_num = args.start_level + src_idx * args.max_extra + (extra - 1)
            result = add_pickups_to_level(prev_data, 1, rng)
            if result is None:
                print(f"  L{level_num}: L{src_lvl} +{extra} = SKIPPED (no room for pickup)")
                break  # can't add more pickups to this level
            all_levels[str(level_num)] = serialize_level(result)
            total = base_pickups + extra
            print(f"  L{level_num}: L{src_lvl} +{extra} = {total} pickups")
            prev_data = result
            created += 1

    with open(LEVELS_PATH, "w") as f:
        json.dump(all_levels, f)

    print(f"\nCreated {created} levels")


if __name__ == "__main__":
    main()
