use std::collections::BinaryHeap;
use std::cmp::Reverse;
use crate::real_game::RealSpaceAceGame;

const CELL_SIZE: f32 = 10.0;
const INFLATION_RADIUS: f32 = 38.0;
/// Fallback inflation used for pickups unreachable under the standard
/// INFLATION_RADIUS — some levels (e.g. level 6) have corridors narrower
/// than 2×38=76px that the standard inflation seals off on both sides.
/// Reduced to the ship's actual collision radius (36.5px, real_physics.rs):
/// any corridor the ship could physically pass through stays unblocked,
/// and any corridor too narrow for the ship remains blocked. No false
/// safety margin — just geometric truth.
const TIGHT_INFLATION_RADIUS: f32 = 36.5;
/// A pickup's distance field must reach at least this many cells to be
/// considered "usable"; otherwise we fall back to the tight grid.
const MIN_REACHABLE_CELLS: usize = 100;

/// Raw difficulty metrics computed from pathfinder analysis.
/// All values are unnormalized; normalization happens in Python.
#[derive(Debug, Clone, Default)]
pub struct DifficultyMetrics {
    // Tier 1: Static geometry
    pub num_walls: u32,
    pub wall_density: f64,
    pub num_pickups: u32,
    pub pickup_spread: f64,

    // Tier 2: Pathfinder-derived
    pub total_route_length: f64,
    pub detour_ratio: f64,
    pub bottleneck_clearance: f64,  // minimum wall distance along optimal route

    // Tier 3: Physics-informed
    pub upward_travel: f64,          // total upward distance along route
    pub upward_travel_tight: f64,    // upward distance in tight corridors (<73px clearance)
    pub maneuver_count: u32,         // sharp turns in tight spaces
    pub worst_maneuver_angle: f64,   // radians, largest turn angle at a tight point

    // Map dimensions
    pub map_width: f64,
    pub map_height: f64,
}

fn point_to_segment_dist_sq(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-10 {
        let dx2 = px - x1;
        let dy2 = py - y1;
        return dx2 * dx2 + dy2 * dy2;
    }
    let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj_x = x1 + t * dx;
    let proj_y = y1 + t * dy;
    let dx2 = px - proj_x;
    let dy2 = py - proj_y;
    dx2 * dx2 + dy2 * dy2
}

// (dr, dc, cost) — cardinal=10, diagonal=14 (≈10*√2) for proper Euclidean-like gradients
const NEIGHBORS: [(i32, i32, i32); 8] = [
    (-1, 0, 10), (1, 0, 10), (0, -1, 10), (0, 1, 10),
    (-1, -1, 14), (-1, 1, 14), (1, -1, 14), (1, 1, 14),
];

pub struct PathfinderGrid {
    rows: usize,
    cols: usize,
    min_x: f32,
    min_y: f32,
    blocked: Vec<bool>,
    /// Secondary grid built with TIGHT_INFLATION_RADIUS, used only for
    /// pickups whose distance field came up empty under standard inflation.
    blocked_tight: Vec<bool>,
    distance_fields: Vec<Vec<i32>>, // one per pickup
    /// True if pickup i's distance field was built on `blocked_tight` rather
    /// than `blocked`. Routing to such a pickup must snap the ship's cell
    /// against the tight grid, or it gets teleported out of the corridor.
    uses_tight_grid: Vec<bool>,
    pickup_dist_matrix: Vec<Vec<i32>>,
    pickup_coords: Vec<(f32, f32)>,
    pub total_pickups: usize,
}

impl PathfinderGrid {
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    pub fn build(game: &RealSpaceAceGame) -> Self {
        let bounds = game.get_map_bounds();
        let min_x = bounds.min_x;
        let min_y = bounds.min_y;
        let cols = ((bounds.max_x - bounds.min_x) / CELL_SIZE) as usize + 1;
        let rows = ((bounds.max_y - bounds.min_y) / CELL_SIZE) as usize + 1;

        let map_lines = game.get_map_lines();
        let blocked = Self::build_blocked_grid(rows, cols, min_x, min_y, &map_lines, INFLATION_RADIUS);
        let blocked_tight = Self::build_blocked_grid(rows, cols, min_x, min_y, &map_lines, TIGHT_INFLATION_RADIUS);

        let pickup_positions = game.get_pickup_positions();
        let pickup_coords: Vec<(f32, f32)> = pickup_positions.iter().map(|&(x, y, _)| (x, y)).collect();
        let total_pickups = pickup_coords.len();

        let mut distance_fields = Vec::with_capacity(total_pickups);
        let mut uses_tight_grid = Vec::with_capacity(total_pickups);
        for &(px, py) in &pickup_coords {
            // Build on both grids and pick whichever reaches more cells —
            // checking a single grid and thresholding doesn't distinguish a
            // pickup that's truly walled off from one that happens to sit in
            // a large local cavity. The tight grid reaches strictly more
            // cells whenever there's a sub-76px corridor the standard grid
            // can't traverse, so "more cells reached" is the right signal.
            let df_std = Self::dijkstra_from(rows, cols, min_x, min_y, &blocked, px, py);
            let df_tight = Self::dijkstra_from(rows, cols, min_x, min_y, &blocked_tight, px, py);
            let reach_std = df_std.iter().filter(|&&d| d >= 0).count();
            let reach_tight = df_tight.iter().filter(|&&d| d >= 0).count();
            // Require a meaningful improvement before switching grids (avoids
            // jitter from 1-2 extra cells) and a floor to catch truly-sealed
            // pickups that reach only their own cavity.
            let tight_wins = reach_tight > reach_std + MIN_REACHABLE_CELLS;
            if tight_wins {
                distance_fields.push(df_tight);
                uses_tight_grid.push(true);
            } else {
                distance_fields.push(df_std);
                uses_tight_grid.push(false);
            }
        }

        // Pickup-to-pickup distance matrix. To read distance_fields[j] we need
        // pickup i's cell snapped against the grid j's field was built on —
        // otherwise a tight pickup at world coords inside standard-inflation
        // looks up -1 in a standard field and the pair looks unreachable.
        let mut pickup_dist_matrix = vec![vec![0i32; total_pickups]; total_pickups];
        for i in 0..total_pickups {
            let (px, py) = pickup_coords[i];
            for j in 0..total_pickups {
                if i == j { continue; }
                let grid_j = if uses_tight_grid[j] { &blocked_tight } else { &blocked };
                let (pr, pc) = Self::to_grid_unblocked(rows, cols, min_x, min_y, grid_j, px, py);
                let d = distance_fields[j][pr * cols + pc];
                pickup_dist_matrix[i][j] = if d >= 0 { d } else { i32::MAX / 2 };
            }
        }

        PathfinderGrid {
            rows, cols, min_x, min_y, blocked, blocked_tight,
            distance_fields, uses_tight_grid,
            pickup_dist_matrix, pickup_coords, total_pickups,
        }
    }

    /// Choose the blocked grid a given pickup's distance field was built on.
    /// Routing / ship-cell snapping must agree with the field, or the ship
    /// gets teleported out of a narrow corridor when navigating to a tight pickup.
    fn blocked_for_pickup(&self, pickup_idx: usize) -> &[bool] {
        if self.uses_tight_grid[pickup_idx] { &self.blocked_tight } else { &self.blocked }
    }

    /// Snap the ship's world coordinates to a grid cell valid for reading
    /// pickup `pickup_idx`'s distance field.
    fn ship_cell_for_pickup(&self, pickup_idx: usize, ship_x: f32, ship_y: f32) -> (usize, usize) {
        Self::to_grid_unblocked(
            self.rows, self.cols, self.min_x, self.min_y,
            self.blocked_for_pickup(pickup_idx), ship_x, ship_y)
    }

    fn to_grid(rows: usize, cols: usize, min_x: f32, min_y: f32, x: f32, y: f32) -> (usize, usize) {
        let c = ((x - min_x) / CELL_SIZE) as i32;
        let r = ((y - min_y) / CELL_SIZE) as i32;
        let r = r.clamp(0, rows as i32 - 1) as usize;
        let c = c.clamp(0, cols as i32 - 1) as usize;
        (r, c)
    }

    fn to_grid_unblocked(rows: usize, cols: usize, min_x: f32, min_y: f32,
                          blocked: &[bool], x: f32, y: f32) -> (usize, usize) {
        let (r, c) = Self::to_grid(rows, cols, min_x, min_y, x, y);
        if !blocked[r * cols + c] {
            return (r, c);
        }
        for radius in 1..10 {
            for dr in -radius..=radius {
                for dc in -radius..=radius {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nr < rows as i32 && nc >= 0 && nc < cols as i32 {
                        let nr = nr as usize;
                        let nc = nc as usize;
                        if !blocked[nr * cols + nc] {
                            return (nr, nc);
                        }
                    }
                }
            }
        }
        (r, c)
    }

    fn build_blocked_grid(rows: usize, cols: usize, min_x: f32, min_y: f32,
                           map_lines: &[(f32, f32, f32, f32)], inflation: f32) -> Vec<bool> {
        let mut blocked = vec![false; rows * cols];
        let inf_sq = inflation * inflation;

        for &(x1, y1, x2, y2) in map_lines {
            let (r_min, c_min) = Self::to_grid(rows, cols, min_x, min_y,
                x1.min(x2) - inflation, y1.min(y2) - inflation);
            let (r_max, c_max) = Self::to_grid(rows, cols, min_x, min_y,
                x1.max(x2) + inflation, y1.max(y2) + inflation);

            for r in r_min..=r_max {
                for c in c_min..=c_max {
                    let idx = r * cols + c;
                    if blocked[idx] { continue; }
                    let cx = min_x + (c as f32 + 0.5) * CELL_SIZE;
                    let cy = min_y + (r as f32 + 0.5) * CELL_SIZE;
                    if point_to_segment_dist_sq(cx, cy, x1, y1, x2, y2) < inf_sq {
                        blocked[idx] = true;
                    }
                }
            }
        }
        blocked
    }

    fn dijkstra_from(rows: usize, cols: usize, min_x: f32, min_y: f32,
                     blocked: &[bool], start_x: f32, start_y: f32) -> Vec<i32> {
        let mut dist = vec![-1i32; rows * cols];
        let (sr, sc) = Self::to_grid(rows, cols, min_x, min_y, start_x, start_y);

        dist[sr * cols + sc] = 0;
        // Min-heap of (cost, row, col)
        let mut heap = BinaryHeap::new();

        if blocked[sr * cols + sc] {
            // Start cell is inside a wall's inflation zone. Seed from the nearest
            // unblocked cells within a wider search radius (up to ~100px / CELL_SIZE cells).
            let max_radius: i32 = 10; // 10 cells * 10px = 100px search
            let mut found = false;
            for radius in 1..=max_radius {
                for dr in -radius..=radius {
                    for dc_val in -radius..=radius {
                        if dr.abs() != radius && dc_val.abs() != radius { continue; }
                        let nr = sr as i32 + dr;
                        let nc = sc as i32 + dc_val;
                        if nr >= 0 && nr < rows as i32 && nc >= 0 && nc < cols as i32 {
                            let nr = nr as usize;
                            let nc = nc as usize;
                            let idx = nr * cols + nc;
                            if !blocked[idx] {
                                // Cost proportional to distance from actual start
                                let cell_dist = ((dr.abs().max(dc_val.abs())) as i32) * 10;
                                dist[idx] = cell_dist;
                                heap.push(Reverse((cell_dist, nr, nc)));
                                found = true;
                            }
                        }
                    }
                }
                if found { break; }
            }
        } else {
            heap.push(Reverse((0, sr, sc)));
        }

        while let Some(Reverse((d, r, c))) = heap.pop() {
            if d > dist[r * cols + c] { continue; }
            for &(dr, dc, cost) in &NEIGHBORS {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr >= 0 && nr < rows as i32 && nc >= 0 && nc < cols as i32 {
                    let nr = nr as usize;
                    let nc = nc as usize;
                    let idx = nr * cols + nc;
                    if !blocked[idx] {
                        let nd = d + cost;
                        if dist[idx] == -1 || nd < dist[idx] {
                            dist[idx] = nd;
                            heap.push(Reverse((nd, nr, nc)));
                        }
                    }
                }
            }
        }
        dist
    }

    pub fn get_nearest_pickup_info(&self, ship_x: f32, ship_y: f32, collected: &[bool]) -> (f64, f64, f64) {
        let (sr_std, sc_std) = Self::to_grid_unblocked(
            self.rows, self.cols, self.min_x, self.min_y, &self.blocked, ship_x, ship_y);
        // Only compute the tight snap if any pickup actually needs it.
        let has_tight = self.uses_tight_grid.iter().any(|&t| t);
        let (sr_tight, sc_tight) = if has_tight {
            Self::to_grid_unblocked(
                self.rows, self.cols, self.min_x, self.min_y, &self.blocked_tight, ship_x, ship_y)
        } else {
            (sr_std, sc_std)
        };
        let pickup_cell = |i: usize| -> (usize, usize) {
            if self.uses_tight_grid[i] { (sr_tight, sc_tight) } else { (sr_std, sc_std) }
        };

        let uncollected: Vec<usize> = (0..self.total_pickups).filter(|&i| !collected[i]).collect();
        if uncollected.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        // Greedy TSP: pick first pickup that minimizes total remaining path
        let mut best_total = i64::MAX;
        let mut best_idx = uncollected[0];

        for &i in &uncollected {
            let (sr, sc) = pickup_cell(i);
            let d_to_i = self.distance_fields[i][sr * self.cols + sc];
            if d_to_i < 0 { continue; }
            let remaining: Vec<usize> = uncollected.iter().copied().filter(|&j| j != i).collect();
            let total = d_to_i as i64 + self.greedy_remaining_cost(i, &remaining);
            if total < best_total {
                best_total = total;
                best_idx = i;
            }
        }

        let (sr, sc) = pickup_cell(best_idx);
        let dist_to_target = self.distance_fields[best_idx][sr * self.cols + sc];
        if dist_to_target < 0 {
            return (0.0, 0.0, 0.0);
        }

        // Distances are now in weighted units (cardinal=10, diagonal=14)
        let world_dist = dist_to_target as f64 * CELL_SIZE as f64 / 10.0;

        // Close range: use direct Euclidean direction (BFS gradient is noisy near pickup)
        let (tx, ty) = self.pickup_coords[best_idx];
        let euclid_dx = (tx - ship_x) as f64;
        let euclid_dy = (ty - ship_y) as f64;
        let euclid_dist = (euclid_dx * euclid_dx + euclid_dy * euclid_dy).sqrt();
        if euclid_dist < 150.0 && euclid_dist > 0.0 {
            // Use Euclidean distance for proximity (pixel-level precision vs 10px grid)
            return (euclid_dist, euclid_dx / euclid_dist, euclid_dy / euclid_dist);
        }

        // Long range: gradient descent on BFS distance field (respects walls)
        let df = &self.distance_fields[best_idx];
        let mut best_nr = sr;
        let mut best_nc = sc;
        let mut best_nd = df[sr * self.cols + sc];

        for &(dr, dc, _cost) in &NEIGHBORS {
            let nr = sr as i32 + dr;
            let nc = sc as i32 + dc;
            if nr >= 0 && nr < self.rows as i32 && nc >= 0 && nc < self.cols as i32 {
                let nr = nr as usize;
                let nc = nc as usize;
                let nd = df[nr * self.cols + nc];
                if nd >= 0 && nd < best_nd {
                    best_nd = nd;
                    best_nr = nr;
                    best_nc = nc;
                }
            }
        }

        let dir_x = (best_nc as f64) - (sc as f64);
        let dir_y = (best_nr as f64) - (sr as f64);
        let mag = (dir_x * dir_x + dir_y * dir_y).sqrt();
        if mag > 0.0 {
            (world_dist, dir_x / mag, dir_y / mag)
        } else {
            (world_dist, 0.0, 0.0)
        }
    }

    /// Look-ahead route tangent. Traces `look_ahead_cells` grid steps along
    /// the BFS gradient toward the greedy-TSP-best first pickup and returns
    /// the normalized direction from the ship to that projected point.
    ///
    /// This is the direction you should *be moving* to stay on the optimal
    /// route, not the direction to the next pickup — the two differ when the
    /// route bends around a wall (the tangent hugs the corridor; the pickup
    /// direction cuts through the wall). For the velocity/orientation terms
    /// of the heuristic, the tangent is what rewards expert swoop/corner play.
    ///
    /// Falls back to the direct pickup vector when within Euclidean collection
    /// range (the tangent concept degenerates at the pickup itself) or when
    /// the distance field is unreachable.
    ///
    /// Returns (dir_x, dir_y, ok) — `ok` is false if no uncollected pickup
    /// reachable; callers should fall back to zero velocity bonus in that case.
    pub fn get_route_tangent(&self, ship_x: f32, ship_y: f32, collected: &[bool], look_ahead_cells: usize) -> (f64, f64, bool) {
        let uncollected: Vec<usize> = (0..self.total_pickups).filter(|&i| !collected[i]).collect();
        if uncollected.is_empty() { return (0.0, 0.0, false); }

        // Pick the same target as get_nearest_pickup_info / get_remaining_route_length
        // so the tangent is consistent with the distance/route signals.
        let (sr_std, sc_std) = Self::to_grid_unblocked(
            self.rows, self.cols, self.min_x, self.min_y, &self.blocked, ship_x, ship_y);
        let has_tight = self.uses_tight_grid.iter().any(|&t| t);
        let (sr_tight, sc_tight) = if has_tight {
            Self::to_grid_unblocked(
                self.rows, self.cols, self.min_x, self.min_y, &self.blocked_tight, ship_x, ship_y)
        } else {
            (sr_std, sc_std)
        };
        let pickup_cell = |i: usize| -> (usize, usize) {
            if self.uses_tight_grid[i] { (sr_tight, sc_tight) } else { (sr_std, sc_std) }
        };

        let mut best_total = i64::MAX;
        let mut best_idx = uncollected[0];
        for &i in &uncollected {
            let (sr, sc) = pickup_cell(i);
            let d_to_i = self.distance_fields[i][sr * self.cols + sc];
            if d_to_i < 0 { continue; }
            let remaining: Vec<usize> = uncollected.iter().copied().filter(|&j| j != i).collect();
            let total = d_to_i as i64 + self.greedy_remaining_cost(i, &remaining);
            if total < best_total { best_total = total; best_idx = i; }
        }

        let (tx, ty) = self.pickup_coords[best_idx];
        let (sr, sc) = pickup_cell(best_idx);
        let df = &self.distance_fields[best_idx];

        // Close range: tangent degenerates — use direct-to-pickup.
        let euclid_dx = (tx - ship_x) as f64;
        let euclid_dy = (ty - ship_y) as f64;
        let euclid_dist = (euclid_dx * euclid_dx + euclid_dy * euclid_dy).sqrt();
        if euclid_dist < 120.0 {
            if euclid_dist > 1.0 {
                return (euclid_dx / euclid_dist, euclid_dy / euclid_dist, true);
            }
            return (0.0, 0.0, true);
        }

        if df[sr * self.cols + sc] < 0 {
            // Unreachable: fall back to direct.
            if euclid_dist > 1.0 {
                return (euclid_dx / euclid_dist, euclid_dy / euclid_dist, true);
            }
            return (0.0, 0.0, true);
        }

        // Trace N steps down the gradient.
        let mut r = sr;
        let mut c = sc;
        for _ in 0..look_ahead_cells {
            let d = df[r * self.cols + c];
            if d <= 0 { break; }
            let mut best_nr = r;
            let mut best_nc = c;
            let mut best_nd = d;
            for &(dr, dc, _cost) in &NEIGHBORS {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr >= 0 && nr < self.rows as i32 && nc >= 0 && nc < self.cols as i32 {
                    let nr = nr as usize;
                    let nc = nc as usize;
                    let nd = df[nr * self.cols + nc];
                    if nd >= 0 && nd < best_nd { best_nd = nd; best_nr = nr; best_nc = nc; }
                }
            }
            if best_nr == r && best_nc == c { break; }
            r = best_nr; c = best_nc;
        }

        let look_x = self.min_x + (c as f32 + 0.5) * CELL_SIZE;
        let look_y = self.min_y + (r as f32 + 0.5) * CELL_SIZE;
        let dx = (look_x - ship_x) as f64;
        let dy = (look_y - ship_y) as f64;
        let mag = (dx * dx + dy * dy).sqrt();
        if mag > 1.0 { (dx / mag, dy / mag, true) }
        else if euclid_dist > 1.0 { (euclid_dx / euclid_dist, euclid_dy / euclid_dist, true) }
        else { (0.0, 0.0, true) }
    }

    /// Greedy TSP length of the remaining tour from the ship, in world px.
    /// Computes: min over first-pickup choices of (ship→i + greedy(i, rest)).
    /// Returns 0.0 if no pickups remain. Used by the MCTS heuristic for route
    /// quality beyond the nearest pickup.
    pub fn get_remaining_route_length(&self, ship_x: f32, ship_y: f32, collected: &[bool]) -> f64 {
        let uncollected: Vec<usize> = (0..self.total_pickups).filter(|&i| !collected[i]).collect();
        if uncollected.is_empty() { return 0.0; }

        let (sr_std, sc_std) = Self::to_grid_unblocked(
            self.rows, self.cols, self.min_x, self.min_y, &self.blocked, ship_x, ship_y);
        let has_tight = self.uses_tight_grid.iter().any(|&t| t);
        let (sr_tight, sc_tight) = if has_tight {
            Self::to_grid_unblocked(
                self.rows, self.cols, self.min_x, self.min_y, &self.blocked_tight, ship_x, ship_y)
        } else {
            (sr_std, sc_std)
        };
        let pickup_cell = |i: usize| -> (usize, usize) {
            if self.uses_tight_grid[i] { (sr_tight, sc_tight) } else { (sr_std, sc_std) }
        };

        let mut best_total = i64::MAX;
        for &i in &uncollected {
            let (sr, sc) = pickup_cell(i);
            let d_to_i = self.distance_fields[i][sr * self.cols + sc];
            if d_to_i < 0 { continue; }
            let remaining: Vec<usize> = uncollected.iter().copied().filter(|&j| j != i).collect();
            let total = d_to_i as i64 + self.greedy_remaining_cost(i, &remaining);
            if total < best_total { best_total = total; }
        }
        if best_total >= i64::MAX / 4 { return 0.0; }
        best_total as f64 * CELL_SIZE as f64 / 10.0
    }

    pub fn get_debug_path(&self, ship_x: f32, ship_y: f32, collected: &[bool]) -> Vec<(f32, f32)> {
        let (sr, sc) = Self::to_grid_unblocked(
            self.rows, self.cols, self.min_x, self.min_y, &self.blocked, ship_x, ship_y);

        let uncollected: Vec<usize> = (0..self.total_pickups).filter(|&i| !collected[i]).collect();
        if uncollected.is_empty() { return vec![]; }

        // Find best target (same as get_nearest_pickup_info)
        let mut best_total = i64::MAX;
        let mut best_idx = uncollected[0];
        for &i in &uncollected {
            let d_to_i = self.distance_fields[i][sr * self.cols + sc];
            if d_to_i < 0 { continue; }
            let remaining: Vec<usize> = uncollected.iter().copied().filter(|&j| j != i).collect();
            let total = d_to_i as i64 + self.greedy_remaining_cost(i, &remaining);
            if total < best_total {
                best_total = total;
                best_idx = i;
            }
        }

        // Trace gradient descent
        let df = &self.distance_fields[best_idx];
        let mut path = Vec::new();
        let mut r = sr;
        let mut c = sc;
        let mut visited = std::collections::HashSet::new();
        loop {
            let x = self.min_x + (c as f32 + 0.5) * CELL_SIZE;
            let y = self.min_y + (r as f32 + 0.5) * CELL_SIZE;
            path.push((x, y));
            let d = df[r * self.cols + c];
            if d <= 0 { break; }
            if !visited.insert((r, c)) { break; }

            let mut best_nr = r;
            let mut best_nc = c;
            let mut best_nd = d;
            for &(dr, dc, _cost) in &NEIGHBORS {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr >= 0 && nr < self.rows as i32 && nc >= 0 && nc < self.cols as i32 {
                    let nr = nr as usize;
                    let nc = nc as usize;
                    let nd = df[nr * self.cols + nc];
                    if nd >= 0 && nd < best_nd {
                        best_nd = nd;
                        best_nr = nr;
                        best_nc = nc;
                    }
                }
            }
            if best_nr == r && best_nc == c { break; }
            r = best_nr;
            c = best_nc;
        }
        path
    }

    /// Returns (target_pickup_index, target_x, target_y, path_dist, euclidean_dist, dir_x, dir_y)
    pub fn get_debug_target_info(&self, ship_x: f32, ship_y: f32, collected: &[bool]) -> (i32, f32, f32, f64, f64, f64, f64) {
        let uncollected: Vec<usize> = (0..self.total_pickups).filter(|&i| !collected[i]).collect();
        if uncollected.is_empty() {
            return (-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let mut best_total = i64::MAX;
        let mut best_idx = uncollected[0];
        for &i in &uncollected {
            let (sr_i, sc_i) = self.ship_cell_for_pickup(i, ship_x, ship_y);
            let d_to_i = self.distance_fields[i][sr_i * self.cols + sc_i];
            if d_to_i < 0 { continue; }
            let remaining: Vec<usize> = uncollected.iter().copied().filter(|&j| j != i).collect();
            let total = d_to_i as i64 + self.greedy_remaining_cost(i, &remaining);
            if total < best_total {
                best_total = total;
                best_idx = i;
            }
        }

        let (sr, sc) = self.ship_cell_for_pickup(best_idx, ship_x, ship_y);
        let (tx, ty) = self.pickup_coords[best_idx];
        let dist_to_target = self.distance_fields[best_idx][sr * self.cols + sc];
        let world_dist = if dist_to_target >= 0 { dist_to_target as f64 * CELL_SIZE as f64 / 10.0 } else { -1.0 };
        let euclidean = (((ship_x - tx) as f64).powi(2) + ((ship_y - ty) as f64).powi(2)).sqrt();

        // Close range: use direct Euclidean direction (BFS gradient is noisy near pickup)
        let euclid_dx = (tx - ship_x) as f64;
        let euclid_dy = (ty - ship_y) as f64;
        if euclidean < 150.0 && euclidean > 0.0 {
            return (best_idx as i32, tx, ty, world_dist, euclidean, euclid_dx / euclidean, euclid_dy / euclidean);
        }

        // Long range: gradient descent on BFS distance field
        let df = &self.distance_fields[best_idx];
        let mut best_nr = sr;
        let mut best_nc = sc;
        let mut best_nd = if dist_to_target >= 0 { dist_to_target } else { 0 };
        for &(dr, dc, _cost) in &NEIGHBORS {
            let nr = sr as i32 + dr;
            let nc = sc as i32 + dc;
            if nr >= 0 && nr < self.rows as i32 && nc >= 0 && nc < self.cols as i32 {
                let nr = nr as usize;
                let nc = nc as usize;
                let nd = df[nr * self.cols + nc];
                if nd >= 0 && nd < best_nd {
                    best_nd = nd;
                    best_nr = nr;
                    best_nc = nc;
                }
            }
        }
        let dir_x = (best_nc as f64) - (sc as f64);
        let dir_y = (best_nr as f64) - (sr as f64);
        let mag = (dir_x * dir_x + dir_y * dir_y).sqrt();
        let (dx, dy) = if mag > 0.0 { (dir_x / mag, dir_y / mag) } else { (0.0, 0.0) };

        (best_idx as i32, tx, ty, world_dist, euclidean, dx, dy)
    }

    /// Returns (path_distance, dir_x, dir_y) toward a specific pickup index.
    /// Like get_nearest_pickup_info but targets a fixed pickup rather than greedy-best.
    pub fn get_distance_to_specific_pickup(&self, ship_x: f32, ship_y: f32, pickup_idx: usize) -> (f64, f64, f64) {
        if pickup_idx >= self.total_pickups {
            return (0.0, 0.0, 0.0);
        }

        let (sr, sc) = self.ship_cell_for_pickup(pickup_idx, ship_x, ship_y);

        let (tx, ty) = self.pickup_coords[pickup_idx];
        let euclid_dx = (tx - ship_x) as f64;
        let euclid_dy = (ty - ship_y) as f64;
        let euclid_dist = (euclid_dx * euclid_dx + euclid_dy * euclid_dy).sqrt();

        let dist_to_target = self.distance_fields[pickup_idx][sr * self.cols + sc];
        if dist_to_target < 0 {
            // BFS unreachable — fall back to Euclidean direction
            if euclid_dist > 0.0 {
                return (euclid_dist, euclid_dx / euclid_dist, euclid_dy / euclid_dist);
            }
            return (0.0, 0.0, 0.0);
        }

        let world_dist = dist_to_target as f64 * CELL_SIZE as f64 / 10.0;

        // Close range: use direct Euclidean direction
        if euclid_dist < 150.0 && euclid_dist > 0.0 {
            return (euclid_dist, euclid_dx / euclid_dist, euclid_dy / euclid_dist);
        }

        // Long range: gradient descent on BFS distance field
        let df = &self.distance_fields[pickup_idx];
        let mut best_nr = sr;
        let mut best_nc = sc;
        let mut best_nd = df[sr * self.cols + sc];

        for &(dr, dc, _cost) in &NEIGHBORS {
            let nr = sr as i32 + dr;
            let nc = sc as i32 + dc;
            if nr >= 0 && nr < self.rows as i32 && nc >= 0 && nc < self.cols as i32 {
                let nr = nr as usize;
                let nc = nc as usize;
                let nd = df[nr * self.cols + nc];
                if nd >= 0 && nd < best_nd {
                    best_nd = nd;
                    best_nr = nr;
                    best_nc = nc;
                }
            }
        }

        let dir_x = (best_nc as f64) - (sc as f64);
        let dir_y = (best_nr as f64) - (sr as f64);
        let mag = (dir_x * dir_x + dir_y * dir_y).sqrt();
        if mag > 0.0 {
            (world_dist, dir_x / mag, dir_y / mag)
        } else {
            (world_dist, 0.0, 0.0)
        }
    }

    /// Trace the full grid path from the ship to a specific pickup via gradient descent
    /// on its precomputed distance field. Returns a list of (x, y) world coordinates.
    pub fn get_path_to_specific_pickup(&self, ship_x: f32, ship_y: f32, pickup_idx: usize) -> Vec<(f32, f32)> {
        if pickup_idx >= self.total_pickups {
            return vec![];
        }

        let (sr, sc) = Self::to_grid_unblocked(
            self.rows, self.cols, self.min_x, self.min_y, &self.blocked, ship_x, ship_y);

        let df = &self.distance_fields[pickup_idx];
        if df[sr * self.cols + sc] < 0 {
            // Unreachable from ship position
            return vec![];
        }

        let mut path = Vec::new();
        let mut r = sr;
        let mut c = sc;
        let mut visited = std::collections::HashSet::new();
        loop {
            let x = self.min_x + (c as f32 + 0.5) * CELL_SIZE;
            let y = self.min_y + (r as f32 + 0.5) * CELL_SIZE;
            path.push((x, y));
            let d = df[r * self.cols + c];
            if d <= 0 { break; }
            if !visited.insert((r, c)) { break; }

            let mut best_nr = r;
            let mut best_nc = c;
            let mut best_nd = d;
            for &(dr, dc, _cost) in &NEIGHBORS {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr >= 0 && nr < self.rows as i32 && nc >= 0 && nc < self.cols as i32 {
                    let nr = nr as usize;
                    let nc = nc as usize;
                    let nd = df[nr * self.cols + nc];
                    if nd >= 0 && nd < best_nd {
                        best_nd = nd;
                        best_nr = nr;
                        best_nc = nc;
                    }
                }
            }
            if best_nr == r && best_nc == c { break; }
            r = best_nr;
            c = best_nc;
        }
        path
    }

    /// Held-Karp exact TSP solver. Returns optimal ordering of uncollected pickups.
    /// Uses bitmask DP: O(n^2 * 2^n), feasible for n <= 20.
    /// Falls back to greedy + 2-opt for n > 20.
    pub fn held_karp_tsp(&self, ship_x: f32, ship_y: f32, collected: &[bool]) -> Vec<usize> {
        let uncollected: Vec<usize> = (0..self.total_pickups)
            .filter(|&i| !collected[i])
            .collect();
        let n = uncollected.len();

        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return uncollected;
        }

        // Compute ship-to-pickup distances
        let (sr, sc) = Self::to_grid_unblocked(
            self.rows, self.cols, self.min_x, self.min_y, &self.blocked, ship_x, ship_y);
        let mut ship_to: Vec<i64> = Vec::with_capacity(n);
        for &i in &uncollected {
            let d = self.distance_fields[i][sr * self.cols + sc];
            ship_to.push(if d >= 0 { d as i64 } else { i64::MAX / 4 });
        }

        // Build local distance matrix for the uncollected subset
        let mut dist = vec![vec![i64::MAX / 4; n]; n];
        for a in 0..n {
            for b in 0..n {
                if a != b {
                    let d = self.pickup_dist_matrix[uncollected[a]][uncollected[b]];
                    dist[a][b] = if d < i32::MAX / 2 { d as i64 } else { i64::MAX / 4 };
                }
            }
        }

        if n > 20 {
            // Fallback: greedy + 2-opt
            return self.greedy_two_opt_tsp(ship_x, ship_y, &uncollected, &ship_to, &dist);
        }

        // Held-Karp DP
        let full_mask: usize = (1 << n) - 1;
        let inf: i64 = i64::MAX / 4;

        // dp[mask][i] = min cost to visit the pickups in `mask`, ending at pickup i
        // mask is a bitmask over the local indices 0..n
        let mut dp = vec![vec![inf; n]; 1 << n];
        let mut parent = vec![vec![usize::MAX; n]; 1 << n];

        // Base case: start from ship, visit pickup i first
        for i in 0..n {
            dp[1 << i][i] = ship_to[i];
        }

        // Fill DP
        for mask in 1..=full_mask {
            for last in 0..n {
                if mask & (1 << last) == 0 { continue; }
                if dp[mask][last] >= inf { continue; }

                for next in 0..n {
                    if mask & (1 << next) != 0 { continue; }
                    let new_mask = mask | (1 << next);
                    let new_cost = dp[mask][last] + dist[last][next];
                    if new_cost < dp[new_mask][next] {
                        dp[new_mask][next] = new_cost;
                        parent[new_mask][next] = last;
                    }
                }
            }
        }

        // Find best ending pickup
        let mut best_cost = inf;
        let mut best_last = 0;
        for i in 0..n {
            if dp[full_mask][i] < best_cost {
                best_cost = dp[full_mask][i];
                best_last = i;
            }
        }

        // Reconstruct path
        let mut order = Vec::with_capacity(n);
        let mut mask = full_mask;
        let mut cur = best_last;
        for _ in 0..n {
            if cur >= n { break; } // unreachable pickup in path
            order.push(uncollected[cur]);
            let prev = parent[mask][cur];
            mask ^= 1 << cur;
            cur = prev;
        }
        order.reverse();
        order
    }

    fn greedy_two_opt_tsp(&self, _ship_x: f32, _ship_y: f32,
                           uncollected: &[usize], ship_to: &[i64],
                           dist: &[Vec<i64>]) -> Vec<usize> {
        let n = uncollected.len();

        // Greedy nearest-neighbor starting from cheapest first pickup
        let mut best_start = 0;
        let mut best_start_cost = i64::MAX;
        for i in 0..n {
            if ship_to[i] < best_start_cost {
                best_start_cost = ship_to[i];
                best_start = i;
            }
        }

        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut visited = vec![false; n];
        order.push(best_start);
        visited[best_start] = true;

        for _ in 1..n {
            let last = *order.last().unwrap();
            let mut best_next = 0;
            let mut best_d = i64::MAX;
            for j in 0..n {
                if !visited[j] && dist[last][j] < best_d {
                    best_d = dist[last][j];
                    best_next = j;
                }
            }
            order.push(best_next);
            visited[best_next] = true;
        }

        // 2-opt improvement
        let route_cost = |ord: &[usize]| -> i64 {
            let mut c = ship_to[ord[0]];
            for w in ord.windows(2) {
                c = c.saturating_add(dist[w[0]][w[1]]);
            }
            c
        };

        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..n - 1 {
                for j in i + 1..n {
                    let old_cost = route_cost(&order);
                    order[i..=j].reverse();
                    let new_cost = route_cost(&order);
                    if new_cost < old_cost {
                        improved = true;
                    } else {
                        order[i..=j].reverse(); // revert
                    }
                }
            }
        }

        order.iter().map(|&i| uncollected[i]).collect()
    }

    /// Check if all pickups are reachable from (spawn_x, spawn_y).
    /// Returns (all_reachable, per_pickup_distances). Unreachable pickups get f64::INFINITY.
    pub fn validate_reachability(&self, spawn_x: f32, spawn_y: f32) -> (bool, Vec<f64>) {
        // Read each pickup's own distance field at the spawn cell snapped
        // against the grid that field was built on. A tight-corridor pickup
        // whose field lives on the reduced-inflation grid is reachable iff
        // that field has a finite value at spawn, regardless of whether the
        // standard-inflation grid can route there.
        let mut all_reachable = true;
        let mut pickup_dists = Vec::with_capacity(self.total_pickups);
        for i in 0..self.total_pickups {
            let (sr, sc) = self.ship_cell_for_pickup(i, spawn_x, spawn_y);
            let d = self.distance_fields[i][sr * self.cols + sc];
            if d < 0 {
                all_reachable = false;
                pickup_dists.push(f64::INFINITY);
            } else {
                pickup_dists.push(d as f64 * CELL_SIZE as f64 / 10.0);
            }
        }
        (all_reachable, pickup_dists)
    }

    pub fn get_pickup_coords(&self) -> &[(f32, f32)] {
        &self.pickup_coords
    }

    /// Analyze level difficulty metrics. Returns a DifficultyMetrics struct
    /// containing all raw sub-scores for the difficulty algorithm.
    /// `ship_x, ship_y` should be the spawn position.
    pub fn analyze_difficulty(&self, ship_x: f32, ship_y: f32,
                               map_lines: &[(f32, f32, f32, f32)]) -> DifficultyMetrics {
        let collected = vec![false; self.total_pickups];

        // --- 1. Build clearance field: actual min wall distance per unblocked cell ---
        let clearance = self.build_clearance_field(map_lines);

        // --- 2. Compute TSP order and route ---
        let tsp_order = self.held_karp_tsp(ship_x, ship_y, &collected);
        if tsp_order.is_empty() || self.total_pickups == 0 {
            return DifficultyMetrics::default();
        }

        // Trace full path along TSP order: ship -> pickup[0] -> pickup[1] -> ...
        let mut full_path_cells: Vec<(usize, usize)> = Vec::new();
        let mut segment_bfs_dists: Vec<f64> = Vec::new();
        let mut segment_euclid_dists: Vec<f64> = Vec::new();

        // Ship to first pickup
        let (sr, sc) = Self::to_grid_unblocked(
            self.rows, self.cols, self.min_x, self.min_y, &self.blocked, ship_x, ship_y);
        {
            let path = self.get_path_to_specific_pickup(ship_x, ship_y, tsp_order[0]);
            for &(wx, wy) in &path {
                let (r, c) = Self::to_grid(self.rows, self.cols, self.min_x, self.min_y, wx, wy);
                full_path_cells.push((r, c));
            }
            let d = self.distance_fields[tsp_order[0]][sr * self.cols + sc];
            let bfs_dist = if d >= 0 { d as f64 * CELL_SIZE as f64 / 10.0 } else { 0.0 };
            let (tx, ty) = self.pickup_coords[tsp_order[0]];
            let euclid = (((ship_x - tx) as f64).powi(2) + ((ship_y - ty) as f64).powi(2)).sqrt();
            segment_bfs_dists.push(bfs_dist);
            segment_euclid_dists.push(euclid);
        }

        // Between consecutive pickups
        for w in tsp_order.windows(2) {
            let from_idx = w[0];
            let to_idx = w[1];
            let (fx, fy) = self.pickup_coords[from_idx];
            let path = self.get_path_to_specific_pickup(fx, fy, to_idx);
            for &(wx, wy) in &path {
                let (r, c) = Self::to_grid(self.rows, self.cols, self.min_x, self.min_y, wx, wy);
                full_path_cells.push((r, c));
            }
            let (fr, fc) = Self::to_grid(self.rows, self.cols, self.min_x, self.min_y, fx, fy);
            let d = self.distance_fields[to_idx][fr * self.cols + fc];
            let bfs_dist = if d >= 0 { d as f64 * CELL_SIZE as f64 / 10.0 } else { 0.0 };
            let (tx, ty) = self.pickup_coords[to_idx];
            let euclid = (((fx - tx) as f64).powi(2) + ((fy - ty) as f64).powi(2)).sqrt();
            segment_bfs_dists.push(bfs_dist);
            segment_euclid_dists.push(euclid);
        }

        // --- 3. Metrics from route ---

        // Total route length
        let total_route_length: f64 = segment_bfs_dists.iter().sum();

        // Detour ratio
        let detour_ratio = if !segment_bfs_dists.is_empty() {
            let ratios: Vec<f64> = segment_bfs_dists.iter().zip(segment_euclid_dists.iter())
                .map(|(&b, &e)| if e > 1.0 { b / e } else { 1.0 })
                .collect();
            ratios.iter().sum::<f64>() / ratios.len() as f64
        } else {
            1.0
        };

        // Bottleneck clearance along route
        let mut min_clearance: f32 = f32::MAX;
        for &(r, c) in &full_path_cells {
            let idx = r * self.cols + c;
            if idx < clearance.len() && clearance[idx] < min_clearance {
                min_clearance = clearance[idx];
            }
        }
        if min_clearance == f32::MAX { min_clearance = 200.0; }

        // Upward travel distance (gravity penalty)
        let mut upward_travel: f64 = 0.0;
        let mut upward_travel_tight: f64 = 0.0;
        for w in full_path_cells.windows(2) {
            let (r1, _c1) = w[0];
            let (r2, _c2) = w[1];
            // In screen coords, lower row = higher on screen = upward
            if r2 < r1 {
                let dy = (r1 - r2) as f64 * CELL_SIZE as f64;
                upward_travel += dy;
                // Check if this segment is in a tight area
                let idx = r2 * self.cols + w[1].1;
                let cl = if idx < clearance.len() { clearance[idx] } else { 200.0 };
                if cl < 73.0 { // 2x ship radius
                    upward_travel_tight += dy;
                }
            }
        }

        // Maneuver points: sharp turns in tight corridors
        let mut maneuver_count: u32 = 0;
        let mut worst_maneuver_angle: f64 = 0.0;
        if full_path_cells.len() >= 3 {
            // Sample every 3 cells to avoid noise
            let step = 3;
            let mut i = 0;
            while i + 2 * step < full_path_cells.len() {
                let (r0, c0) = full_path_cells[i];
                let (r1, c1) = full_path_cells[i + step];
                let (r2, c2) = full_path_cells[i + 2 * step];

                let dx1 = c1 as f64 - c0 as f64;
                let dy1 = r1 as f64 - r0 as f64;
                let dx2 = c2 as f64 - c1 as f64;
                let dy2 = r2 as f64 - r1 as f64;

                let mag1 = (dx1 * dx1 + dy1 * dy1).sqrt();
                let mag2 = (dx2 * dx2 + dy2 * dy2).sqrt();

                if mag1 > 0.1 && mag2 > 0.1 {
                    let dot = dx1 * dx2 + dy1 * dy2;
                    let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
                    let angle = cos_angle.acos(); // radians, 0 = straight, PI = u-turn

                    if angle > std::f64::consts::FRAC_PI_4 { // > 45 degrees
                        // Check clearance at turn point
                        let idx = r1 * self.cols + c1;
                        let cl = if idx < clearance.len() { clearance[idx] } else { 200.0 };
                        if cl < 73.0 { // 2x ship radius
                            maneuver_count += 1;
                            if angle > worst_maneuver_angle {
                                worst_maneuver_angle = angle;
                            }
                        }
                    }
                }
                i += step;
            }
        }

        // --- 4. Static geometry metrics ---

        // Wall density
        let total_wall_length: f64 = map_lines.iter().map(|&(x1, y1, x2, y2)| {
            (((x2 - x1) as f64).powi(2) + ((y2 - y1) as f64).powi(2)).sqrt()
        }).sum();
        let map_w = (self.cols as f64) * CELL_SIZE as f64;
        let map_h = (self.rows as f64) * CELL_SIZE as f64;
        let map_area = map_w * map_h;
        let wall_density = if map_area > 0.0 { total_wall_length / map_area } else { 0.0 };

        // Vertex/line counts
        let num_walls = map_lines.len() as u32;

        // Pickup spread: mean pairwise distance
        let pickup_spread = if self.total_pickups >= 2 {
            let mut total = 0.0f64;
            let mut count = 0u32;
            for i in 0..self.total_pickups {
                for j in (i+1)..self.total_pickups {
                    let (x1, y1) = self.pickup_coords[i];
                    let (x2, y2) = self.pickup_coords[j];
                    total += (((x2 - x1) as f64).powi(2) + ((y2 - y1) as f64).powi(2)).sqrt();
                    count += 1;
                }
            }
            total / count as f64
        } else {
            0.0
        };

        DifficultyMetrics {
            // Tier 1: Static geometry
            num_walls,
            wall_density,
            num_pickups: self.total_pickups as u32,
            pickup_spread,

            // Tier 2: Pathfinder-derived
            total_route_length,
            detour_ratio,
            bottleneck_clearance: min_clearance as f64,

            // Tier 3: Physics-informed
            upward_travel,
            upward_travel_tight,
            maneuver_count,
            worst_maneuver_angle,

            // Map dimensions
            map_width: map_w,
            map_height: map_h,
        }
    }

    /// Build a clearance field: for each unblocked cell, stores the actual
    /// minimum distance to the nearest wall segment. Blocked cells get 0.0.
    fn build_clearance_field(&self, map_lines: &[(f32, f32, f32, f32)]) -> Vec<f32> {
        let mut clearance = vec![0.0f32; self.rows * self.cols];

        for r in 0..self.rows {
            for c in 0..self.cols {
                let idx = r * self.cols + c;
                if self.blocked[idx] {
                    continue;
                }
                let cx = self.min_x + (c as f32 + 0.5) * CELL_SIZE;
                let cy = self.min_y + (r as f32 + 0.5) * CELL_SIZE;

                let mut min_dist_sq = f32::MAX;
                for &(x1, y1, x2, y2) in map_lines {
                    let d = point_to_segment_dist_sq(cx, cy, x1, y1, x2, y2);
                    if d < min_dist_sq {
                        min_dist_sq = d;
                    }
                }
                clearance[idx] = min_dist_sq.sqrt();
            }
        }
        clearance
    }

    fn greedy_remaining_cost(&self, start_idx: usize, remaining: &[usize]) -> i64 {
        let mut cost: i64 = 0;
        let mut current = start_idx;
        let mut left: Vec<usize> = remaining.to_vec();
        while !left.is_empty() {
            let mut best_d = i32::MAX / 2;
            let mut best_j = left[0];
            for &j in &left {
                let d = self.pickup_dist_matrix[current][j];
                if d < best_d {
                    best_d = d;
                    best_j = j;
                }
            }
            if best_d >= i32::MAX / 2 { break; }
            cost += best_d as i64;
            left.retain(|&x| x != best_j);
            current = best_j;
        }
        cost
    }
}
