#!/usr/bin/env python3
"""
SpaceAce procedural map generator.
Generates playable maps ranked by difficulty, output in the same flat-array
JSON format consumed by the Rust engine (src/real_map_parser.rs).

Usage:
    uv run python generate_maps.py --count 20 --visualize
    uv run python generate_maps.py --count 10 --merge --strategy room
"""

import argparse
import json
import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants (must match Rust engine)
# ---------------------------------------------------------------------------
CELL_SIZE = 10.0
INFLATION_RADIUS = 35.0
SHIP_COLLISION_RADIUS = 36.5
PICKUP_RADIUS = 10.0
SPAWN_Y_OFFSET = -100.0  # ship spawns 100px above the spawn vertex
MIN_CORRIDOR_WIDTH = 80.0  # safe margin over theoretical 71px minimum


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class GeneratedMap:
    vertices: list  # list of (x, y)
    lines: list     # list of (va, vb) vertex index pairs
    pickups: list   # list of vertex indices
    start_index: int
    bounding_width: float
    bounding_height: float
    difficulty_score: float = 0.0
    strategy_name: str = ""


# ---------------------------------------------------------------------------
# Flat-array serializer (matches parse_spaceace_data in real_map_parser.rs)
# ---------------------------------------------------------------------------
def serialize_map(m: GeneratedMap) -> list:
    """Encode a GeneratedMap into the flat number array the Rust parser expects."""
    arr = [len(m.vertices)]
    for x, y in m.vertices:
        arr.extend([x, y])
    arr.append(len(m.lines))
    for va, vb in m.lines:
        arr.extend([va, vb])
    arr.append(m.start_index)
    arr.append(m.bounding_width)
    arr.append(m.bounding_height)
    arr.append(len(m.pickups))
    for p in m.pickups:
        arr.append(p)
    arr.append(0)  # triangle_count
    return arr


# ---------------------------------------------------------------------------
# Geometry helper (used by corridor-width measurement, NOT pathfinding)
# ---------------------------------------------------------------------------
def point_to_segment_dist_sq(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-10:
        dx2 = px - x1
        dy2 = py - y1
        return dx2 * dx2 + dy2 * dy2
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    dx2 = px - proj_x
    dy2 = py - proj_y
    return dx2 * dx2 + dy2 * dy2


# ---------------------------------------------------------------------------
# Validation engine — delegates to Rust RustPathfinder via PyO3
# ---------------------------------------------------------------------------
import spaceace_rl


class MapValidator:
    """Validates that all pickups are reachable from spawn using the Rust
    pathfinder (BFS on an inflated grid)."""

    def __init__(self, m: GeneratedMap):
        self.m = m
        map_json = json.dumps(serialize_map(m))
        self._pf = spaceace_rl.PyPathfinder.from_map_json(map_json)

    def validate(self):
        """Returns per-pickup world distances if all reachable, else False."""
        m = self.m
        spawn_v = m.vertices[m.start_index]
        spawn_x = spawn_v[0]
        spawn_y = spawn_v[1] + SPAWN_Y_OFFSET
        all_reachable, dists = self._pf.validate_reachability(spawn_x, spawn_y)
        if not all_reachable:
            return False
        return dists

    def compute_distances(self, dists):
        """Return per-pickup distances (already in world units from Rust)."""
        return list(dists)


# ---------------------------------------------------------------------------
# Difficulty scorer
# ---------------------------------------------------------------------------
def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def compute_min_corridor_width(m: GeneratedMap):
    """Estimate minimum passage width by checking distances between
    non-adjacent wall segments."""
    segs = []
    for va, vb in m.lines:
        segs.append((m.vertices[va], m.vertices[vb]))

    if len(segs) < 2:
        return 999.0

    # Build adjacency set
    adj = set()
    for va, vb in m.lines:
        adj.add((va, vb))
        adj.add((vb, va))

    min_dist = 999.0
    for i in range(len(m.lines)):
        va1, vb1 = m.lines[i]
        s1 = segs[i]
        for j in range(i + 1, len(m.lines)):
            va2, vb2 = m.lines[j]
            # Skip segments sharing a vertex (they're connected)
            if va1 == va2 or va1 == vb2 or vb1 == va2 or vb1 == vb2:
                continue
            s2 = segs[j]
            # Check all four point-to-segment distances
            d = min(
                math.sqrt(point_to_segment_dist_sq(s1[0][0], s1[0][1], s2[0][0], s2[0][1], s2[1][0], s2[1][1])),
                math.sqrt(point_to_segment_dist_sq(s1[1][0], s1[1][1], s2[0][0], s2[0][1], s2[1][0], s2[1][1])),
                math.sqrt(point_to_segment_dist_sq(s2[0][0], s2[0][1], s1[0][0], s1[0][1], s1[1][0], s1[1][1])),
                math.sqrt(point_to_segment_dist_sq(s2[1][0], s2[1][1], s1[0][0], s1[0][1], s1[1][0], s1[1][1])),
            )
            if d < min_dist:
                min_dist = d
    return min_dist


def compute_difficulty(m: GeneratedMap, bfs_distances: list) -> float:
    # 1. Corridor width factor
    min_corr = compute_min_corridor_width(m)
    corridor_score = 1.0 - _clamp((min_corr - 71) / (200 - 71))

    # 2. Path complexity: BFS vs euclidean distance ratio
    # Use euclidean distances directly for the ratio to avoid BFS grid quantization
    # artifacts on short distances. BFS is still used for reachability + total length.
    spawn_v = m.vertices[m.start_index]
    spawn_x, spawn_y = spawn_v[0], spawn_v[1] + SPAWN_Y_OFFSET
    total_bfs = 0.0
    total_eucl = 0.0
    for i, pi in enumerate(m.pickups):
        px, py = m.vertices[pi]
        eucl = math.sqrt((px - spawn_x) ** 2 + (py - spawn_y) ** 2)
        bfs_d = bfs_distances[i] if bfs_distances[i] > 0 else eucl
        total_bfs += bfs_d
        total_eucl += max(eucl, 1.0)
    # BFS on a grid inflates distances by ~sqrt(2)/2 even for straight-line paths,
    # so subtract 1.3 baseline before scaling
    ratio = total_bfs / max(total_eucl, 1.0)
    complexity_score = _clamp((ratio - 1.3) / 3.0)

    # 3. Pickup count factor
    pickup_score = _clamp((len(m.pickups) - 1) / 8.0)

    # 4. Total path length (euclidean-based to avoid grid inflation)
    length_score = _clamp(total_eucl / 5000.0)

    # 5. Gravity danger: pickups above ship spawn position require upward thrust
    above_count = sum(1 for pi in m.pickups if m.vertices[pi][1] < spawn_y)
    gravity_score = _clamp(above_count / max(len(m.pickups), 1))

    # 6. Wall density (interior walls relative to map area)
    area = m.bounding_width * m.bounding_height
    total_wall_len = sum(
        math.sqrt((m.vertices[va][0] - m.vertices[vb][0]) ** 2 +
                   (m.vertices[va][1] - m.vertices[vb][1]) ** 2)
        for va, vb in m.lines
    )
    # Subtract perimeter (boundary walls don't add interior difficulty)
    perimeter = 2 * (m.bounding_width + m.bounding_height)
    interior_wall_len = max(0, total_wall_len - perimeter)
    density_score = _clamp(interior_wall_len / max(math.sqrt(area), 1.0) * 0.5)

    score = (
        corridor_score * 0.25 +
        complexity_score * 0.20 +
        pickup_score * 0.15 +
        length_score * 0.15 +
        gravity_score * 0.15 +
        density_score * 0.10
    )
    return round(score, 4)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def _add_rect_walls(vertices, lines, x, y, w, h):
    """Add a rectangle as 4 vertices + 4 wall lines. Returns corner indices."""
    base = len(vertices)
    vertices.append((x, y))
    vertices.append((x + w, y))
    vertices.append((x + w, y + h))
    vertices.append((x, y + h))
    lines.append((base, base + 1))
    lines.append((base + 1, base + 2))
    lines.append((base + 2, base + 3))
    lines.append((base + 3, base))
    return base, base + 1, base + 2, base + 3


def _add_polygon(vertices, lines, cx, cy, radius, sides):
    """Add a regular polygon obstacle. Returns list of vertex indices."""
    base = len(vertices)
    idxs = []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        vertices.append((x, y))
        idxs.append(base + i)
    for i in range(sides):
        lines.append((idxs[i], idxs[(i + 1) % sides]))
    return idxs


# ---------------------------------------------------------------------------
# Strategy A: Room-and-Corridor
# ---------------------------------------------------------------------------
class RoomCorridorGenerator:
    name = "room"

    def __init__(self, rng: random.Random, target_difficulty: float):
        self.rng = rng
        self.td = target_difficulty

    def generate(self) -> GeneratedMap:
        rng = self.rng
        td = self.td

        # Scale parameters by difficulty
        num_rooms = int(3 + td * 5)  # 3-8
        corridor_width = 160 - td * 80  # 160-80
        room_min = int(200 - td * 80)   # 200-120
        room_max = int(400 - td * 120)  # 400-280
        num_pickups = max(1, int(1 + td * 8))  # 1-9

        rooms = []  # (x, y, w, h)
        for _ in range(num_rooms * 20):
            if len(rooms) >= num_rooms:
                break
            w = rng.randint(room_min, room_max)
            h = rng.randint(room_min, room_max)
            x = rng.randint(100, 1500)
            y = rng.randint(200, 1500)  # leave space for spawn offset
            margin = corridor_width + 40
            ok = True
            for rx, ry, rw, rh in rooms:
                if not (x + w + margin < rx or rx + rw + margin < x or
                        y + h + margin < ry or ry + rh + margin < y):
                    ok = False
                    break
            if ok:
                rooms.append((x, y, w, h))

        if len(rooms) < 2:
            rooms = [(100, 300, 300, 300), (600, 300, 300, 300)]

        centers = [(x + w / 2, y + h / 2) for x, y, w, h in rooms]
        edges = self._mst(centers)

        # Build geometry using a grid-stamp approach:
        # Rasterize rooms and corridors onto a coarse boolean grid,
        # then extract boundary wall segments from the grid edges.
        # This naturally creates openings where rooms meet corridors.
        GRID = 10.0  # cell size for rasterization

        # Compute bounds
        all_coords = []
        for x, y, w, h in rooms:
            all_coords.extend([(x, y), (x + w, y + h)])
        for i, j in edges:
            all_coords.extend([centers[i], centers[j]])
        xs = [c[0] for c in all_coords]
        ys = [c[1] for c in all_coords]
        gx0 = min(xs) - corridor_width - 50
        gy0 = min(ys) - 200  # extra for spawn offset
        gx1 = max(xs) + corridor_width + 50
        gy1 = max(ys) + corridor_width + 50
        gcols = int((gx1 - gx0) / GRID) + 2
        grows = int((gy1 - gy0) / GRID) + 2

        # Carve open cells
        open_cells = set()

        def carve_rect(rx, ry, rw, rh):
            c0 = int((rx - gx0) / GRID)
            r0 = int((ry - gy0) / GRID)
            c1 = int((rx + rw - gx0) / GRID)
            r1 = int((ry + rh - gy0) / GRID)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if 0 <= rr < grows and 0 <= cc < gcols:
                        open_cells.add((rr, cc))

        # Carve rooms
        for x, y, w, h in rooms:
            carve_rect(x, y, w, h)

        # Carve L-shaped corridors
        hw = corridor_width / 2
        for i, j in edges:
            cx1, cy1 = centers[i]
            cx2, cy2 = centers[j]
            mid_x, mid_y = cx2, cy1
            # Horizontal leg
            x_lo, x_hi = min(cx1, mid_x), max(cx1, mid_x)
            carve_rect(x_lo - hw, mid_y - hw, x_hi - x_lo + 2 * hw, corridor_width)
            # Vertical leg
            y_lo, y_hi = min(mid_y, cy2), max(mid_y, cy2)
            carve_rect(mid_x - hw, y_lo - hw, corridor_width, y_hi - y_lo + 2 * hw)

        # Extract boundary wall segments from grid edges
        # A wall exists where an open cell borders a closed cell
        vertices = []
        lines = []
        edge_set = set()  # deduplicate

        def add_wall_seg(wx1, wy1, wx2, wy2):
            key = (wx1, wy1, wx2, wy2)
            if key in edge_set:
                return
            edge_set.add(key)
            b = len(vertices)
            vertices.append((wx1, wy1))
            vertices.append((wx2, wy2))
            lines.append((b, b + 1))

        for (r, c) in open_cells:
            wx = gx0 + c * GRID
            wy = gy0 + r * GRID
            # Check each of the 4 neighbors
            if (r - 1, c) not in open_cells:  # top edge
                add_wall_seg(wx, wy, wx + GRID, wy)
            if (r + 1, c) not in open_cells:  # bottom edge
                add_wall_seg(wx, wy + GRID, wx + GRID, wy + GRID)
            if (r, c - 1) not in open_cells:  # left edge
                add_wall_seg(wx, wy, wx, wy + GRID)
            if (r, c + 1) not in open_cells:  # right edge
                add_wall_seg(wx + GRID, wy, wx + GRID, wy + GRID)

        # Merge collinear wall segments to reduce vertex/line count
        vertices, lines = _merge_collinear_walls(vertices, lines)

        # Place spawn in first room
        sx, sy, sw, sh = rooms[0]
        spawn_idx = len(vertices)
        vertices.append((sx + sw / 2, sy + sh - 50))

        # Place pickups in other rooms
        pickup_idxs = []
        for k in range(num_pickups):
            room = rooms[(k + 1) % len(rooms)]
            rx, ry, rw, rh = room
            px = rx + rw * (0.3 + rng.random() * 0.4)
            py = ry + rh * (0.3 + rng.random() * 0.4)
            pi = len(vertices)
            vertices.append((px, py))
            pickup_idxs.append(pi)

        all_x = [v[0] for v in vertices]
        all_y = [v[1] for v in vertices]
        bw = max(all_x) - min(all_x) + 100
        bh = max(all_y) - min(all_y) + 200

        return GeneratedMap(
            vertices=vertices, lines=lines, pickups=pickup_idxs,
            start_index=spawn_idx, bounding_width=bw, bounding_height=bh,
            strategy_name=self.name,
        )

    def _mst(self, centers):
        n = len(centers)
        if n <= 1:
            return []
        in_tree = [False] * n
        in_tree[0] = True
        edges = []
        for _ in range(n - 1):
            best_d = float("inf")
            best_e = None
            for i in range(n):
                if not in_tree[i]:
                    continue
                for j in range(n):
                    if in_tree[j]:
                        continue
                    d = math.hypot(centers[i][0] - centers[j][0],
                                   centers[i][1] - centers[j][1])
                    if d < best_d:
                        best_d = d
                        best_e = (i, j)
            if best_e:
                edges.append(best_e)
                in_tree[best_e[1]] = True
        return edges


def _merge_collinear_walls(vertices, lines):
    """Merge collinear adjacent wall segments to reduce geometry count."""
    # Group horizontal segments by y-coordinate, vertical by x-coordinate
    h_segs = {}  # y -> list of (x_start, x_end)
    v_segs = {}  # x -> list of (y_start, y_end)

    for va, vb in lines:
        x1, y1 = vertices[va]
        x2, y2 = vertices[vb]
        if abs(y1 - y2) < 0.01:  # horizontal
            y = round(y1, 1)
            h_segs.setdefault(y, []).append((min(x1, x2), max(x1, x2)))
        elif abs(x1 - x2) < 0.01:  # vertical
            x = round(x1, 1)
            v_segs.setdefault(x, []).append((min(y1, y2), max(y1, y2)))

    new_verts = []
    new_lines = []

    def merge_intervals(intervals):
        intervals.sort()
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            if start <= merged[-1][1] + 0.01:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    for y, segs in h_segs.items():
        for x1, x2 in merge_intervals(segs):
            b = len(new_verts)
            new_verts.append((x1, y))
            new_verts.append((x2, y))
            new_lines.append((b, b + 1))

    for x, segs in v_segs.items():
        for y1, y2 in merge_intervals(segs):
            b = len(new_verts)
            new_verts.append((x, y1))
            new_verts.append((x, y2))
            new_lines.append((b, b + 1))

    return new_verts, new_lines


# ---------------------------------------------------------------------------
# Strategy B: Maze
# ---------------------------------------------------------------------------
class MazeGenerator:
    name = "maze"

    def __init__(self, rng: random.Random, target_difficulty: float):
        self.rng = rng
        self.td = target_difficulty

    def generate(self) -> GeneratedMap:
        rng = self.rng
        td = self.td

        cols = int(3 + td * 4)  # 3-7
        rows = int(3 + td * 3)  # 3-6
        cell_w = int(200 - td * 60)  # 200-140
        cell_h = int(200 - td * 60)
        num_pickups = max(1, int(1 + td * 7))
        wall_removal = 0.3 * (1 - td)  # 30%-0%

        # Generate maze using recursive backtracking
        # h_walls[r][c] = True means wall on top of cell (r,c)
        # v_walls[r][c] = True means wall on left of cell (r,c)
        h_walls = [[True] * cols for _ in range(rows + 1)]
        v_walls = [[True] * (cols + 1) for _ in range(rows)]

        visited = [[False] * cols for _ in range(rows)]
        stack = [(0, 0)]
        visited[0][0] = True

        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                    neighbors.append((nr, nc, dr, dc))
            if not neighbors:
                stack.pop()
                continue
            nr, nc, dr, dc = rng.choice(neighbors)
            # Remove wall between (r,c) and (nr,nc)
            if dr == -1:
                h_walls[r][c] = False
            elif dr == 1:
                h_walls[r + 1][c] = False
            elif dc == -1:
                v_walls[r][c] = False
            elif dc == 1:
                v_walls[r][c + 1] = False
            visited[nr][nc] = True
            stack.append((nr, nc))

        # Remove extra walls for easier variants
        if wall_removal > 0:
            interior_h = [(r, c) for r in range(1, rows) for c in range(cols) if h_walls[r][c]]
            interior_v = [(r, c) for r in range(rows) for c in range(1, cols) if v_walls[r][c]]
            all_removable = interior_h + interior_v
            rng.shuffle(all_removable)
            to_remove = int(len(all_removable) * wall_removal)
            for idx in range(to_remove):
                item = all_removable[idx]
                # Determine if h or v wall
                if item in interior_h:
                    h_walls[item[0]][item[1]] = False
                else:
                    v_walls[item[0]][item[1]] = False

        # Convert maze to vertices and wall segments
        offset_x = 100.0
        offset_y = 250.0  # extra space for spawn offset above
        vertices = []
        lines = []

        # Create vertex grid
        v_grid = {}  # (gr, gc) -> vertex index
        for gr in range(rows + 1):
            for gc in range(cols + 1):
                v_grid[(gr, gc)] = len(vertices)
                vertices.append((offset_x + gc * cell_w, offset_y + gr * cell_h))

        # Add horizontal walls
        for r in range(rows + 1):
            for c in range(cols):
                if h_walls[r][c]:
                    lines.append((v_grid[(r, c)], v_grid[(r, c + 1)]))

        # Add vertical walls
        for r in range(rows):
            for c in range(cols + 1):
                if v_walls[r][c]:
                    lines.append((v_grid[(r, c)], v_grid[(r + 1, c)]))

        # Spawn in top-left cell
        spawn_idx = len(vertices)
        vertices.append((offset_x + cell_w * 0.5, offset_y + cell_h * 0.5 + 100))

        # Place pickups spread across cells
        pickup_idxs = []
        available_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != (0, 0)]
        rng.shuffle(available_cells)
        for k in range(min(num_pickups, len(available_cells))):
            cr, cc_ = available_cells[k]
            px = offset_x + (cc_ + 0.5) * cell_w
            py = offset_y + (cr + 0.5) * cell_h
            pi = len(vertices)
            vertices.append((px, py))
            pickup_idxs.append(pi)

        if not pickup_idxs:
            pi = len(vertices)
            vertices.append((offset_x + cell_w * 1.5, offset_y + cell_h * 0.5))
            pickup_idxs.append(pi)

        bw = cols * cell_w + 200
        bh = rows * cell_h + 400

        return GeneratedMap(
            vertices=vertices, lines=lines, pickups=pickup_idxs,
            start_index=spawn_idx, bounding_width=bw, bounding_height=bh,
            strategy_name=self.name,
        )


# ---------------------------------------------------------------------------
# Strategy C: Cave / Asteroid Field
# ---------------------------------------------------------------------------
class CaveGenerator:
    name = "cave"

    def __init__(self, rng: random.Random, target_difficulty: float):
        self.rng = rng
        self.td = target_difficulty

    def generate(self) -> GeneratedMap:
        rng = self.rng
        td = self.td

        map_w = int(800 + td * 800)   # 800-1600
        map_h = int(800 + td * 600)   # 800-1400
        num_obstacles = int(3 + td * 12)  # 3-15
        obstacle_radius = int(30 + td * 40)  # 30-70
        min_gap = MIN_CORRIDOR_WIDTH + 10
        num_pickups = max(1, int(1 + td * 7))
        sides = rng.randint(5, 8)

        vertices = []
        lines = []

        # Outer boundary
        _add_rect_walls(vertices, lines, 0, 0, map_w, map_h)

        # Place obstacles with Poisson-like rejection sampling
        obstacles = []  # (cx, cy, r)
        margin = 100
        for _ in range(num_obstacles * 30):
            if len(obstacles) >= num_obstacles:
                break
            r = obstacle_radius + rng.randint(-10, 10)
            cx = rng.uniform(margin + r, map_w - margin - r)
            cy = rng.uniform(margin + r, map_h - margin - r)
            # Check spacing with existing obstacles
            ok = True
            for ox, oy, or_ in obstacles:
                dist = math.hypot(cx - ox, cy - oy)
                if dist < r + or_ + min_gap:
                    ok = False
                    break
            if ok:
                obstacles.append((cx, cy, r))
                _add_polygon(vertices, lines, cx, cy, r, sides)

        # Spawn point (center top area, leaving room for y-100 offset)
        spawn_idx = len(vertices)
        vertices.append((map_w / 2, 150))

        # Place pickups in gaps between obstacles
        pickup_idxs = []
        for _ in range(num_pickups * 20):
            if len(pickup_idxs) >= num_pickups:
                break
            px = rng.uniform(margin, map_w - margin)
            py = rng.uniform(margin, map_h - margin)
            # Must be far enough from all obstacles and walls
            ok = True
            for ox, oy, or_ in obstacles:
                if math.hypot(px - ox, py - oy) < or_ + INFLATION_RADIUS + 10:
                    ok = False
                    break
            # Must be far from outer walls
            if px < INFLATION_RADIUS + 10 or px > map_w - INFLATION_RADIUS - 10:
                ok = False
            if py < INFLATION_RADIUS + 10 or py > map_h - INFLATION_RADIUS - 10:
                ok = False
            if ok:
                pi = len(vertices)
                vertices.append((px, py))
                pickup_idxs.append(pi)

        if not pickup_idxs:
            pi = len(vertices)
            vertices.append((map_w / 2, map_h / 2))
            pickup_idxs.append(pi)

        return GeneratedMap(
            vertices=vertices, lines=lines, pickups=pickup_idxs,
            start_index=spawn_idx, bounding_width=float(map_w), bounding_height=float(map_h),
            strategy_name=self.name,
        )


# ---------------------------------------------------------------------------
# Strategy D: Gauntlet (linear winding path)
# Uses same grid-stamp approach as RoomCorridorGenerator for reliable walls.
# ---------------------------------------------------------------------------
class GauntletGenerator:
    name = "gauntlet"

    def __init__(self, rng: random.Random, target_difficulty: float):
        self.rng = rng
        self.td = target_difficulty

    def generate(self) -> GeneratedMap:
        rng = self.rng
        td = self.td

        path_width = 160 - td * 70   # 160-90
        num_segments = int(4 + td * 6)  # 4-10
        num_pickups = max(1, int(1 + td * 6))

        hw = path_width / 2

        # Generate waypoints for a serpentine path
        waypoints = [(300.0, 300.0)]
        direction = 0  # 0=right, 1=down, 2=left
        seg_len_base = int(300 - td * 80)  # 300-220

        for _ in range(num_segments):
            last = waypoints[-1]
            seg_len = seg_len_base + rng.randint(-30, 30)
            if direction == 0:
                waypoints.append((last[0] + seg_len, last[1]))
                direction = 1
            elif direction == 1:
                waypoints.append((last[0], last[1] + seg_len))
                direction = 2
            elif direction == 2:
                waypoints.append((last[0] - seg_len, last[1]))
                direction = 1

        # Rasterize path onto grid and extract boundary walls
        GRID = 10.0
        all_pts = waypoints
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        gx0 = min(xs) - hw - 100
        gy0 = min(ys) - 200
        gx1 = max(xs) + hw + 100
        gy1 = max(ys) + hw + 100
        gcols = int((gx1 - gx0) / GRID) + 2
        grows = int((gy1 - gy0) / GRID) + 2

        open_cells = set()

        def carve_rect(rx, ry, rw, rh):
            c0 = int((rx - gx0) / GRID)
            r0 = int((ry - gy0) / GRID)
            c1 = int((rx + rw - gx0) / GRID)
            r1 = int((ry + rh - gy0) / GRID)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if 0 <= rr < grows and 0 <= cc < gcols:
                        open_cells.add((rr, cc))

        # Carve path segments
        for i in range(len(waypoints) - 1):
            x1, y1 = waypoints[i]
            x2, y2 = waypoints[i + 1]
            rx = min(x1, x2) - hw
            ry = min(y1, y2) - hw
            rw = abs(x2 - x1) + 2 * hw
            rh = abs(y2 - y1) + 2 * hw
            carve_rect(rx, ry, rw, rh)

        # Extract boundary walls
        vertices = []
        lines = []
        edge_set = set()

        def add_wall_seg(wx1, wy1, wx2, wy2):
            key = (wx1, wy1, wx2, wy2)
            if key in edge_set:
                return
            edge_set.add(key)
            b = len(vertices)
            vertices.append((wx1, wy1))
            vertices.append((wx2, wy2))
            lines.append((b, b + 1))

        for (r, c) in open_cells:
            wx = gx0 + c * GRID
            wy = gy0 + r * GRID
            if (r - 1, c) not in open_cells:
                add_wall_seg(wx, wy, wx + GRID, wy)
            if (r + 1, c) not in open_cells:
                add_wall_seg(wx, wy + GRID, wx + GRID, wy + GRID)
            if (r, c - 1) not in open_cells:
                add_wall_seg(wx, wy, wx, wy + GRID)
            if (r, c + 1) not in open_cells:
                add_wall_seg(wx + GRID, wy, wx + GRID, wy + GRID)

        vertices, lines = _merge_collinear_walls(vertices, lines)

        # Spawn at start of path
        spawn_idx = len(vertices)
        sx, sy = waypoints[0]
        vertices.append((sx, sy + 100))  # +100 because spawn offsets by -100

        # Place pickups along the path
        pickup_idxs = []
        total_waypoints = len(waypoints)
        for k in range(num_pickups):
            t = (k + 1) / (num_pickups + 1)
            seg_f = t * (total_waypoints - 1)
            seg_i = int(seg_f)
            seg_t = seg_f - seg_i
            if seg_i >= total_waypoints - 1:
                seg_i = total_waypoints - 2
                seg_t = 1.0
            px = waypoints[seg_i][0] + seg_t * (waypoints[seg_i + 1][0] - waypoints[seg_i][0])
            py = waypoints[seg_i][1] + seg_t * (waypoints[seg_i + 1][1] - waypoints[seg_i][1])
            pi = len(vertices)
            vertices.append((px, py))
            pickup_idxs.append(pi)

        if not pickup_idxs:
            pi = len(vertices)
            vertices.append(waypoints[-1])
            pickup_idxs.append(pi)

        all_x = [v[0] for v in vertices]
        all_y = [v[1] for v in vertices]
        bw = max(all_x) - min(all_x) + 200
        bh = max(all_y) - min(all_y) + 300

        return GeneratedMap(
            vertices=vertices, lines=lines, pickups=pickup_idxs,
            start_index=spawn_idx, bounding_width=bw, bounding_height=bh,
            strategy_name=self.name,
        )


# ---------------------------------------------------------------------------
# ASCII visualization
# ---------------------------------------------------------------------------
def ascii_preview(m: GeneratedMap, width=80, height=30) -> str:
    all_x = [v[0] for v in m.vertices]
    all_y = [v[1] for v in m.vertices]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    range_x = max_x - min_x or 1
    range_y = max_y - min_y or 1

    grid = [[' '] * width for _ in range(height)]

    def to_grid(x, y):
        gc = int((x - min_x) / range_x * (width - 1))
        gr = int((y - min_y) / range_y * (height - 1))
        return max(0, min(height - 1, gr)), max(0, min(width - 1, gc))

    # Draw walls
    for va, vb in m.lines:
        x1, y1 = m.vertices[va]
        x2, y2 = m.vertices[vb]
        steps = max(int(math.hypot(x2 - x1, y2 - y1) / (range_x / width)), 1)
        for s in range(steps + 1):
            t = s / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            r, c = to_grid(x, y)
            grid[r][c] = '#'

    # Draw pickups
    for pi in m.pickups:
        px, py = m.vertices[pi]
        r, c = to_grid(px, py)
        grid[r][c] = '*'

    # Draw spawn
    sv = m.vertices[m.start_index]
    sr, sc = to_grid(sv[0], sv[1] + SPAWN_Y_OFFSET)
    grid[sr][sc] = 'S'

    return '\n'.join(''.join(row) for row in grid)


# ---------------------------------------------------------------------------
# Strategy E: Simple (curriculum learning starter levels)
# ---------------------------------------------------------------------------
class SimpleGenerator:
    """Generates a large open room with 1 pickup near the spawn.
    Designed as the easiest possible level for curriculum learning —
    the agent just needs to learn basic thrust/rotation to reach a
    nearby pickup with minimal wall danger."""
    name = "simple"

    def __init__(self, rng: random.Random, target_difficulty: float):
        self.rng = rng
        self.td = target_difficulty

    def generate(self) -> GeneratedMap:
        rng = self.rng
        td = self.td

        # Room size: large so walls are far away
        room_w = 500 + rng.randint(0, 200)
        room_h = 400 + rng.randint(0, 200)
        margin = 50.0

        vertices = []
        lines = []
        _add_rect_walls(vertices, lines, margin, margin, room_w, room_h)

        # Spawn near center of room
        cx = margin + room_w / 2
        cy = margin + room_h / 2
        spawn_idx = len(vertices)
        # Spawn vertex — ship appears 100px above this
        vertices.append((cx, cy + 50))

        # Place pickup nearby — distance scales with difficulty
        # td=0 → ~80px away, td=1 → ~250px away
        pickup_dist = 80 + td * 170
        angle = rng.uniform(0, 2 * math.pi)
        # Bias downward slightly so gravity helps at easiest levels
        if td < 0.3:
            angle = rng.uniform(math.pi * 0.25, math.pi * 0.75)  # below spawn
        px = cx + pickup_dist * math.cos(angle)
        py = (cy - 50) + pickup_dist * math.sin(angle)  # relative to ship pos (cy-50)
        # Clamp inside room
        px = max(margin + 50, min(margin + room_w - 50, px))
        py = max(margin + 50, min(margin + room_h - 50, py))

        pickup_idx = len(vertices)
        vertices.append((px, py))

        return GeneratedMap(
            vertices=vertices, lines=lines, pickups=[pickup_idx],
            start_index=spawn_idx,
            bounding_width=room_w + 2 * margin,
            bounding_height=room_h + 2 * margin,
            strategy_name=self.name,
        )


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------
STRATEGIES = {
    'simple': SimpleGenerator,
    'room': RoomCorridorGenerator,
    'maze': MazeGenerator,
    'cave': CaveGenerator,
    'gauntlet': GauntletGenerator,
}


def generate_maps(count=20, seed=42, strategy='all'):
    rng = random.Random(seed)
    strategy_list = list(STRATEGIES.values()) if strategy == 'all' else [STRATEGIES[strategy]]

    maps = []
    attempts = 0
    max_attempts = count * 20

    while len(maps) < count and attempts < max_attempts:
        attempts += 1
        target_difficulty = len(maps) / max(count - 1, 1)
        gen_cls = rng.choice(strategy_list)
        gen = gen_cls(rng, target_difficulty)
        candidate = gen.generate()

        validator = MapValidator(candidate)
        result = validator.validate()
        if not result:
            continue

        bfs_distances = validator.compute_distances(result)
        candidate.difficulty_score = compute_difficulty(candidate, bfs_distances)
        maps.append(candidate)

    maps.sort(key=lambda m: m.difficulty_score)
    return maps


def write_levels(maps, output_path, start_level=1000):
    result = {}
    for i, m in enumerate(maps):
        level_num = start_level + i
        result[str(level_num)] = serialize_map(m)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    return result


def merge_levels(maps, levels_path, start_level=1000):
    try:
        with open(levels_path, 'r') as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {}

    for i, m in enumerate(maps):
        level_num = start_level + i
        existing[str(level_num)] = serialize_map(m)

    with open(levels_path, 'w') as f:
        json.dump(existing, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Generate SpaceAce maps ranked by difficulty')
    parser.add_argument('--count', type=int, default=20, help='Number of maps (default 20)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default 42)')
    parser.add_argument('--output', type=str, default='data/generated_levels.json',
                        help='Output file (default data/generated_levels.json)')
    parser.add_argument('--merge', action='store_true',
                        help='Also merge into data/spaceace_levels.json')
    parser.add_argument('--start-level', type=int, default=1000,
                        help='Starting level number (default 1000)')
    parser.add_argument('--strategy', choices=['simple', 'room', 'maze', 'cave', 'gauntlet', 'all'],
                        default='all', help='Generation strategy (default all)')
    parser.add_argument('--visualize', action='store_true', help='Print ASCII preview of each map')
    args = parser.parse_args()

    print(f'Generating {args.count} maps (seed={args.seed}, strategy={args.strategy})...')
    maps = generate_maps(count=args.count, seed=args.seed, strategy=args.strategy)

    print(f'Generated {len(maps)} valid maps.')
    for i, m in enumerate(maps):
        level = args.start_level + i
        print(f'  Level {level}: {m.strategy_name:10s}  difficulty={m.difficulty_score:.4f}  '
              f'vertices={len(m.vertices)}  walls={len(m.lines)}  pickups={len(m.pickups)}')
        if args.visualize:
            print(ascii_preview(m))
            print()

    write_levels(maps, args.output, start_level=args.start_level)
    print(f'Wrote {len(maps)} levels to {args.output}')

    if args.merge:
        merge_levels(maps, 'data/spaceace_levels.json', start_level=args.start_level)
        print(f'Merged into data/spaceace_levels.json')


if __name__ == '__main__':
    main()
