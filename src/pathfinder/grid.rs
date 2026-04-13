use std::collections::BinaryHeap;
use std::cmp::Reverse;
use crate::real_game::RealSpaceAceGame;

const CELL_SIZE: f32 = 10.0;
const INFLATION_RADIUS: f32 = 35.0;

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
    distance_fields: Vec<Vec<i32>>, // one per pickup
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
        let blocked = Self::build_blocked_grid(rows, cols, min_x, min_y, &map_lines);

        let pickup_positions = game.get_pickup_positions();
        let pickup_coords: Vec<(f32, f32)> = pickup_positions.iter().map(|&(x, y, _)| (x, y)).collect();
        let total_pickups = pickup_coords.len();

        let mut distance_fields = Vec::with_capacity(total_pickups);
        for &(px, py) in &pickup_coords {
            let df = Self::dijkstra_from(rows, cols, min_x, min_y, &blocked, px, py);
            distance_fields.push(df);
        }

        // Pickup-to-pickup distance matrix
        let mut pickup_dist_matrix = vec![vec![0i32; total_pickups]; total_pickups];
        for i in 0..total_pickups {
            let (pr, pc) = Self::to_grid(rows, cols, min_x, min_y,
                                          pickup_coords[i].0, pickup_coords[i].1);
            for j in 0..total_pickups {
                if i == j { continue; }
                let d = distance_fields[j][pr * cols + pc];
                pickup_dist_matrix[i][j] = if d >= 0 { d } else { i32::MAX / 2 };
            }
        }

        PathfinderGrid {
            rows, cols, min_x, min_y, blocked,
            distance_fields, pickup_dist_matrix, pickup_coords, total_pickups,
        }
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
                           map_lines: &[(f32, f32, f32, f32)]) -> Vec<bool> {
        let mut blocked = vec![false; rows * cols];
        let inf_sq = INFLATION_RADIUS * INFLATION_RADIUS;

        for &(x1, y1, x2, y2) in map_lines {
            let (r_min, c_min) = Self::to_grid(rows, cols, min_x, min_y,
                x1.min(x2) - INFLATION_RADIUS, y1.min(y2) - INFLATION_RADIUS);
            let (r_max, c_max) = Self::to_grid(rows, cols, min_x, min_y,
                x1.max(x2) + INFLATION_RADIUS, y1.max(y2) + INFLATION_RADIUS);

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
            for &(dr, dc, cost) in &NEIGHBORS {
                let nr = sr as i32 + dr;
                let nc = sc as i32 + dc;
                if nr >= 0 && nr < rows as i32 && nc >= 0 && nc < cols as i32 {
                    let nr = nr as usize;
                    let nc = nc as usize;
                    let idx = nr * cols + nc;
                    if !blocked[idx] {
                        dist[idx] = cost;
                        heap.push(Reverse((cost, nr, nc)));
                    }
                }
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
        let (sr, sc) = Self::to_grid_unblocked(
            self.rows, self.cols, self.min_x, self.min_y, &self.blocked, ship_x, ship_y);

        let uncollected: Vec<usize> = (0..self.total_pickups).filter(|&i| !collected[i]).collect();
        if uncollected.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        // Greedy TSP: pick first pickup that minimizes total remaining path
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
        let (sr, sc) = Self::to_grid_unblocked(
            self.rows, self.cols, self.min_x, self.min_y, &self.blocked, ship_x, ship_y);

        let uncollected: Vec<usize> = (0..self.total_pickups).filter(|&i| !collected[i]).collect();
        if uncollected.is_empty() {
            return (-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

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

        let (sr, sc) = Self::to_grid_unblocked(
            self.rows, self.cols, self.min_x, self.min_y, &self.blocked, ship_x, ship_y);

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
        let (sr, sc) = Self::to_grid(self.rows, self.cols, self.min_x, self.min_y, spawn_x, spawn_y);

        // If spawn is blocked, nothing is reachable
        if self.blocked[sr * self.cols + sc] {
            let dists = vec![f64::INFINITY; self.total_pickups];
            return (false, dists);
        }

        // Run Dijkstra from spawn
        let dist = Self::dijkstra_from(self.rows, self.cols, self.min_x, self.min_y,
                                        &self.blocked, spawn_x, spawn_y);

        let mut all_reachable = true;
        let mut pickup_dists = Vec::with_capacity(self.total_pickups);
        for i in 0..self.total_pickups {
            let (px, py) = self.pickup_coords[i];
            let (pr, pc) = Self::to_grid(self.rows, self.cols, self.min_x, self.min_y, px, py);
            let d = dist[pr * self.cols + pc];
            if d < 0 {
                all_reachable = false;
                pickup_dists.push(f64::INFINITY);
            } else {
                // Convert from weighted grid units to world units
                pickup_dists.push(d as f64 * CELL_SIZE as f64 / 10.0);
            }
        }
        (all_reachable, pickup_dists)
    }

    pub fn get_pickup_coords(&self) -> &[(f32, f32)] {
        &self.pickup_coords
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
