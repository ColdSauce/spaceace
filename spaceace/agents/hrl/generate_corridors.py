"""Generate corridor training levels for the HRL waypoint pilot.

Produces simple corridors, L-turns, and S-curves with a single pickup at the
far end. These train the pilot on pure flight mechanics without requiring
maze navigation.

IMPORTANT: Ship spawns 100px ABOVE the start vertex (y - 100). All start
vertices must have y >= top_wall_y + 150 to ensure safe spawn clearance.

Usage:
    uv run python -m spaceace.agents.hrl.generate_corridors
"""

import json
import math
import random
import os

# Ship spawns at (vertex_x, vertex_y - 100). We need vertex_y to be
# at least 150px below the top wall to give clearance.
SPAWN_Y_OFFSET = 100
SPAWN_CLEARANCE = 50  # extra clearance beyond spawn offset


def build_level_array(vertices, lines, start_idx, pickup_indices, bounds_w, bounds_h):
    """Build flat level array in SpaceAce format."""
    data = []
    data.append(len(vertices))
    for x, y in vertices:
        data.append(x)
        data.append(y)
    data.append(len(lines))
    for a, b in lines:
        data.append(a)
        data.append(b)
    data.append(start_idx)
    data.append(bounds_w)
    data.append(bounds_h)
    data.append(len(pickup_indices))
    for p in pickup_indices:
        data.append(p)
    data.append(0)  # triangle count
    return data


def straight_corridor(width, length, margin=50):
    """Straight corridor from left to right."""
    cx = margin
    cy = margin  # top wall y
    # Start vertex at cy+150 so ship spawns at cy+50 (50px below top wall, safe)
    start_y = cy + 150

    vertices = [
        (cx, cy),                       # 0: top-left
        (cx + length, cy),              # 1: top-right
        (cx + length, cy + width),      # 2: bottom-right
        (cx, cy + width),               # 3: bottom-left
        (cx + 50, start_y),             # 4: start
        (cx + length - 50, cy + width / 2),  # 5: pickup (center-right)
    ]
    lines = [(0, 1), (1, 2), (2, 3), (3, 0)]
    bw = length + 2 * margin
    bh = cy + width + margin
    return build_level_array(vertices, lines, 4, [5], bw, bh)


def l_turn(width, leg1_len, leg2_len, margin=50):
    """L-shaped corridor: goes right then down."""
    cx = margin
    cy = margin

    vertices = [
        (cx, cy),                                          # 0
        (cx + leg1_len, cy),                               # 1
        (cx + leg1_len, cy + width + leg2_len),            # 2
        (cx + leg1_len - width, cy + width + leg2_len),    # 3
        (cx + leg1_len - width, cy + width),               # 4
        (cx, cy + width),                                  # 5
        (cx + 50, cy + 150),                               # 6: start (spawn at cy+50)
        (cx + leg1_len - width / 2, cy + width + leg2_len - 50),  # 7: pickup
    ]
    lines = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    bw = leg1_len + 2 * margin
    bh = cy + width + leg2_len + margin
    return build_level_array(vertices, lines, 6, [7], bw, bh)


def s_curve(width, seg_len, offset, margin=50):
    """S-shaped corridor: right, then offset down-right, then right."""
    cx = margin
    cy = margin

    mid_x = cx + seg_len
    end_x = mid_x + seg_len
    shift_y = offset

    vertices = [
        (cx, cy),                          # 0
        (mid_x, cy),                       # 1
        (mid_x + width, cy + shift_y),     # 2
        (end_x + width, cy + shift_y),     # 3
        (end_x + width, cy + shift_y + width),  # 4
        (mid_x + width, cy + shift_y + width),  # 5
        (mid_x, cy + width),               # 6
        (cx, cy + width),                  # 7
        (cx + 50, cy + 150),              # 8: start (spawn at cy+50)
        (end_x + width - 50, cy + shift_y + width / 2),  # 9: pickup
    ]
    lines = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)]
    bw = end_x + width + 2 * margin
    bh = cy + shift_y + width + margin
    return build_level_array(vertices, lines, 8, [9], bw, bh)


def room_with_pillar(room_w, room_h, pillar_size, margin=50):
    """Open room with a central pillar obstacle."""
    cx = margin
    cy = margin
    px = cx + room_w / 2 - pillar_size / 2
    py = cy + room_h / 2 - pillar_size / 2

    vertices = [
        (cx, cy),                      # 0
        (cx + room_w, cy),             # 1
        (cx + room_w, cy + room_h),    # 2
        (cx, cy + room_h),            # 3
        (px, py),                      # 4
        (px + pillar_size, py),        # 5
        (px + pillar_size, py + pillar_size),  # 6
        (px, py + pillar_size),        # 7
        (cx + 80, cy + 150),           # 8: start (spawn at cy+50)
        (cx + room_w - 80, cy + room_h / 2),  # 9: pickup (right side, center)
    ]
    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
    ]
    bw = room_w + 2 * margin
    bh = cy + room_h + margin
    return build_level_array(vertices, lines, 8, [9], bw, bh)


def narrow_gap(approach_len, gap_width, total_width, margin=50):
    """Corridor with a narrow gap in the middle."""
    cx = margin
    cy = margin
    mid_x = cx + approach_len
    gap_top = cy + (total_width - gap_width) / 2
    gap_bot = gap_top + gap_width

    vertices = [
        (cx, cy),                                      # 0
        (cx + approach_len * 2, cy),                   # 1
        (cx + approach_len * 2, cy + total_width),     # 2
        (cx, cy + total_width),                        # 3
        (mid_x, cy),                                   # 4: top of gap wall (on top wall)
        (mid_x, gap_top),                              # 5: gap wall ends here
        (mid_x, gap_bot),                              # 6: gap wall starts here
        (mid_x, cy + total_width),                     # 7: bottom of gap wall (on bottom wall)
        (cx + 50, cy + 150),                             # 8: start (spawn at cy+50)
        (cx + approach_len * 2 - 50, cy + total_width / 2),  # 9: pickup
    ]
    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5),  # top wall segment
        (6, 7),  # bottom wall segment
    ]
    bw = approach_len * 2 + 2 * margin
    bh = cy + total_width + margin
    return build_level_array(vertices, lines, 8, [9], bw, bh)


def generate_all_corridors(seed=42):
    """Generate 100 corridor levels with varying difficulty."""
    levels = {}
    idx = 4000

    # 20 straight corridors (easy wide → narrower)
    for i in range(20):
        width = 350 - i * 5  # 350 → 255
        length = 200 + i * 50  # 200 → 1150
        levels[str(idx)] = straight_corridor(width, length)
        idx += 1

    # 20 L-turns
    for i in range(20):
        width = 350 - i * 5  # 350 → 255
        leg1 = 250 + i * 35
        leg2 = 250 + i * 30
        levels[str(idx)] = l_turn(width, leg1, leg2)
        idx += 1

    # 20 S-curves
    for i in range(20):
        width = 350 - i * 5
        seg_len = 200 + i * 25
        offset = 120 + i * 15
        levels[str(idx)] = s_curve(width, seg_len, offset)
        idx += 1

    # 20 rooms with pillars
    for i in range(20):
        room_w = 400 + i * 30
        room_h = 400 + i * 25
        pillar = 50 + i * 10
        levels[str(idx)] = room_with_pillar(room_w, room_h, pillar)
        idx += 1

    # 20 narrow gaps
    for i in range(20):
        approach = 250 + i * 25
        gap = 150 - i * 3  # 150 → 93
        total_w = 350
        levels[str(idx)] = narrow_gap(approach, gap, total_w)
        idx += 1

    return levels


def main():
    levels = generate_all_corridors()

    levels_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "spaceace_levels.json")
    levels_path = os.path.normpath(levels_path)

    with open(levels_path, "r") as f:
        all_levels = json.load(f)

    # Remove old corridor levels (4000+)
    all_levels = {k: v for k, v in all_levels.items() if k.startswith("_") or int(k) < 4000}

    all_levels.update(levels)

    with open(levels_path, "w") as f:
        json.dump(all_levels, f)

    print(f"Generated {len(levels)} corridor levels (4000-{4000 + len(levels) - 1})")
    print(f"Saved to {levels_path}")


if __name__ == "__main__":
    main()
