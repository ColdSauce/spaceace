use std::collections::BinaryHeap;
use std::cmp::Reverse;
use std::f32::consts::PI;

use crate::pathfinder::PathfinderGrid;
use crate::real_game::RealSpaceAceGame;

// --- Constants ---

const CELL_SIZE: f32 = 20.0;
const INFLATION_RADIUS: f32 = 35.0;
const PICKUP_COLLECTION_RADIUS: f32 = 46.5; // ship 36.5 + pickup 10

// Physics constants (must match real_physics.rs)
const GRAVITY: f32 = 100.0;
const THRUST_POWER: f32 = 400.0;
const ROTATION_SPEED: f32 = 4.363323;
const DT: f32 = 1.0 / 60.0;

// Velocity discretization
const SPEED_BUCKETS: [f32; 6] = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0];
const NUM_SPEED_BUCKETS: usize = 6;
const NUM_ANGLE_BUCKETS: usize = 8;
// Total velocity classes: 1 (zero speed) + 5 (nonzero speeds) * 8 (angles) = 41
const NUM_VEL_CLASSES: usize = 1 + (NUM_SPEED_BUCKETS - 1) * NUM_ANGLE_BUCKETS;

// Forward simulation: how many physics frames per transition edge
const SIM_FRAMES_PER_EDGE: u32 = 5;
// Number of rotation samples when building transitions
const NUM_ROTATION_SAMPLES: usize = 5;

// 6 actions matching mcts.rs
const ACTIONS: [[bool; 3]; 6] = [
    [false, false, false], // coast
    [false, false, true],  // thrust
    [true, false, false],  // rotate left
    [true, false, true],   // rotate left + thrust
    [false, true, false],  // rotate right
    [false, true, true],   // rotate right + thrust
];
const NUM_ACTIONS: usize = 6;

// --- Helpers ---

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

/// Classify a velocity vector into a velocity class index (0..NUM_VEL_CLASSES).
/// Class 0 = zero/near-zero speed. Classes 1..40 = (speed_bucket, angle_bucket).
fn classify_velocity(vx: f32, vy: f32) -> usize {
    let speed = (vx * vx + vy * vy).sqrt();
    if speed < 50.0 {
        return 0; // zero-speed class
    }
    // Find speed bucket (1-indexed, skip bucket 0 which is zero-speed)
    let mut speed_idx = 0usize;
    for i in 1..NUM_SPEED_BUCKETS {
        if speed >= (SPEED_BUCKETS[i] + SPEED_BUCKETS[i.saturating_sub(1)]) * 0.5 {
            speed_idx = i;
        }
    }
    if speed_idx == 0 { speed_idx = 1; } // minimum nonzero bucket

    // Angle bucket
    let angle = vy.atan2(vx); // -PI..PI
    let bucket = ((angle + PI) / (2.0 * PI) * NUM_ANGLE_BUCKETS as f32) as usize;
    let angle_idx = bucket.min(NUM_ANGLE_BUCKETS - 1);

    1 + (speed_idx - 1) * NUM_ANGLE_BUCKETS + angle_idx
}

/// Convert a velocity class back to a representative (vx, vy) vector.
fn vel_class_to_vector(class: usize) -> (f32, f32) {
    if class == 0 {
        return (0.0, 0.0);
    }
    let inner = class - 1;
    let speed_idx = inner / NUM_ANGLE_BUCKETS + 1;
    let angle_idx = inner % NUM_ANGLE_BUCKETS;

    let speed = SPEED_BUCKETS[speed_idx.min(NUM_SPEED_BUCKETS - 1)];
    let angle = -PI + (angle_idx as f32 + 0.5) * (2.0 * PI / NUM_ANGLE_BUCKETS as f32);
    (speed * angle.cos(), speed * angle.sin())
}

/// Simulate physics for `frames` steps with given action and initial state.
/// Returns (final_x, final_y, final_vx, final_vy, hit_wall).
/// `blocked` grid is used for simple wall collision check.
fn simulate_physics(
    start_x: f32, start_y: f32,
    start_vx: f32, start_vy: f32,
    start_rotation: f32,
    action: [bool; 3],
    frames: u32,
    blocked: &[bool],
    rows: usize, cols: usize,
    min_x: f32, min_y: f32,
) -> (f32, f32, f32, f32, bool) {
    let mut x = start_x;
    let mut y = start_y;
    let mut vx = start_vx;
    let mut vy = start_vy;
    let mut rotation = start_rotation;

    for _ in 0..frames {
        // Rotation
        if action[0] { rotation -= ROTATION_SPEED * DT; }
        if action[1] { rotation += ROTATION_SPEED * DT; }
        // Thrust
        if action[2] {
            let angle = rotation - PI * 0.5;
            vx += THRUST_POWER * angle.cos() * DT;
            vy += THRUST_POWER * angle.sin() * DT;
        }
        // Gravity
        vy += GRAVITY * DT;
        // Position
        x += vx * DT;
        y += vy * DT;

        // Check if in blocked cell
        let c = ((x - min_x) / CELL_SIZE) as i32;
        let r = ((y - min_y) / CELL_SIZE) as i32;
        if r < 0 || r >= rows as i32 || c < 0 || c >= cols as i32 {
            return (x, y, vx, vy, true);
        }
        if blocked[r as usize * cols + c as usize] {
            return (x, y, vx, vy, true);
        }
    }
    (x, y, vx, vy, false)
}

/// Compact state index: cell_index * NUM_VEL_CLASSES + vel_class
fn state_index(cell_idx: usize, vel_class: usize) -> usize {
    cell_idx * NUM_VEL_CLASSES + vel_class
}

// --- Edge stored in the forward graph ---
#[derive(Clone, Copy)]
struct Edge {
    to_state: u32,
    cost: u16, // frames
}

// --- MomentumPathfinder ---

pub struct MomentumPathfinder {
    // Grid
    rows: usize,
    cols: usize,
    min_x: f32,
    min_y: f32,
    blocked: Vec<bool>,
    num_cells: usize,
    num_states: usize,

    // Forward graph: edges[state_idx] lists outgoing edges
    forward_edges: Vec<Vec<Edge>>,
    // Reverse graph: reverse_edges[state_idx] lists incoming edges (from_state, cost)
    // Only used during build, but we keep it for direction extraction
    reverse_edges: Vec<Vec<Edge>>,

    // Time-to-reach tables: time_to_reach[pickup_idx][state_idx] = frames (u16::MAX = unreachable)
    time_to_reach: Vec<Vec<u16>>,

    // Pickup-to-pickup with velocity awareness
    // pickup_transit[from_pickup][vel_class][to_pickup] = frames
    pickup_transit: Vec<Vec<Vec<u16>>>,
    // Arrival velocity class at target
    pickup_arrival_vel: Vec<Vec<Vec<u8>>>,

    // Pickup data
    pickup_coords: Vec<(f32, f32)>,
    pub total_pickups: usize,

    // Spatial pathfinder for fallback direction queries
    spatial_distance_fields: Vec<Vec<i32>>,
    spatial_rows: usize,
    spatial_cols: usize,
    spatial_min_x: f32,
    spatial_min_y: f32,
}

impl MomentumPathfinder {
    pub fn build(game: &RealSpaceAceGame) -> Self {
        let start_time = std::time::Instant::now();

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

        let num_cells = rows * cols;
        let num_states = num_cells * NUM_VEL_CLASSES;

        eprintln!("[MomentumPathfinder] Grid: {}x{} = {} cells, {} states, {} pickups",
                  rows, cols, num_cells, num_states, total_pickups);

        // --- Phase 2: Build forward transition graph ---
        let (forward_edges, reverse_edges) = Self::build_transition_graph(
            rows, cols, min_x, min_y, &blocked, num_states,
        );
        eprintln!("[MomentumPathfinder] Transition graph built in {:.2}s",
                  start_time.elapsed().as_secs_f64());

        // --- Phase 3: Backward Dijkstra per pickup ---
        let time_to_reach = Self::compute_time_to_reach(
            rows, cols, min_x, min_y, &pickup_coords, &reverse_edges, num_states,
        );
        eprintln!("[MomentumPathfinder] Time-to-reach computed in {:.2}s",
                  start_time.elapsed().as_secs_f64());

        // --- Phase 4: Pickup-to-pickup cost matrix ---
        let (pickup_transit, pickup_arrival_vel) = Self::compute_pickup_transit(
            rows, cols, min_x, min_y, &pickup_coords, &time_to_reach, total_pickups,
        );

        // --- Build spatial distance fields (reuse pathfinder's Dijkstra for direction fallback) ---
        let spatial_pf = PathfinderGrid::build(game);
        // We need to extract the distance fields from PathfinderGrid.
        // Since we can't access private fields, we'll compute our own lightweight version
        // for direction queries. We'll use the spatial pathfinder through the PathfinderKind enum.
        // For now, store empty and use a simpler direction approach.
        let spatial_distance_fields = Vec::new();

        eprintln!("[MomentumPathfinder] Build complete in {:.2}s", start_time.elapsed().as_secs_f64());

        MomentumPathfinder {
            rows, cols, min_x, min_y, blocked, num_cells, num_states,
            forward_edges, reverse_edges,
            time_to_reach,
            pickup_transit, pickup_arrival_vel,
            pickup_coords, total_pickups,
            spatial_distance_fields,
            spatial_rows: spatial_pf.rows(),
            spatial_cols: spatial_pf.cols(),
            spatial_min_x: 0.0,
            spatial_min_y: 0.0,
        }
    }

    fn build_blocked_grid(rows: usize, cols: usize, min_x: f32, min_y: f32,
                           map_lines: &[(f32, f32, f32, f32)]) -> Vec<bool> {
        let mut blocked = vec![false; rows * cols];
        let inf_sq = INFLATION_RADIUS * INFLATION_RADIUS;

        for &(x1, y1, x2, y2) in map_lines {
            let c_min = ((x1.min(x2) - INFLATION_RADIUS - min_x) / CELL_SIZE) as i32;
            let c_max = ((x1.max(x2) + INFLATION_RADIUS - min_x) / CELL_SIZE) as i32;
            let r_min = ((y1.min(y2) - INFLATION_RADIUS - min_y) / CELL_SIZE) as i32;
            let r_max = ((y1.max(y2) + INFLATION_RADIUS - min_y) / CELL_SIZE) as i32;

            let r_min = r_min.max(0) as usize;
            let r_max = (r_max.min(rows as i32 - 1)) as usize;
            let c_min = c_min.max(0) as usize;
            let c_max = (c_max.min(cols as i32 - 1)) as usize;

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

    fn build_transition_graph(
        rows: usize, cols: usize, min_x: f32, min_y: f32,
        blocked: &[bool], num_states: usize,
    ) -> (Vec<Vec<Edge>>, Vec<Vec<Edge>>) {
        let num_cells = rows * cols;
        let mut forward: Vec<Vec<Edge>> = vec![Vec::new(); num_states];
        let mut reverse: Vec<Vec<Edge>> = vec![Vec::new(); num_states];

        // Rotation offsets to sample: 0, ±π/4, ±π/2
        let rotation_offsets: [f32; NUM_ROTATION_SAMPLES] = [0.0, PI / 4.0, -PI / 4.0, PI / 2.0, -PI / 2.0];

        for cell_idx in 0..num_cells {
            if blocked[cell_idx] { continue; }

            let r = cell_idx / cols;
            let c = cell_idx % cols;
            let cx = min_x + (c as f32 + 0.5) * CELL_SIZE;
            let cy = min_y + (r as f32 + 0.5) * CELL_SIZE;

            for vel_class in 0..NUM_VEL_CLASSES {
                let from_state = state_index(cell_idx, vel_class);
                let (base_vx, base_vy) = vel_class_to_vector(vel_class);

                // Base rotation: point in velocity direction (or upward if zero speed)
                let base_rot = if vel_class == 0 {
                    0.0 // pointing up
                } else {
                    base_vy.atan2(base_vx) + PI * 0.5 // rotation convention: 0 = up
                };

                for action_idx in 0..NUM_ACTIONS {
                    let action = ACTIONS[action_idx];
                    let mut best_to_state: Option<u32> = None;

                    // Try multiple rotation samples, keep the best (unique) transitions
                    for &rot_offset in &rotation_offsets {
                        let rot = base_rot + rot_offset;
                        let (fx, fy, fvx, fvy, hit_wall) = simulate_physics(
                            cx, cy, base_vx, base_vy, rot, action,
                            SIM_FRAMES_PER_EDGE, blocked, rows, cols, min_x, min_y,
                        );
                        if hit_wall { continue; }

                        // Snap to discrete state
                        let fc = ((fx - min_x) / CELL_SIZE) as i32;
                        let fr = ((fy - min_y) / CELL_SIZE) as i32;
                        if fr < 0 || fr >= rows as i32 || fc < 0 || fc >= cols as i32 { continue; }
                        let fr = fr as usize;
                        let fc = fc as usize;
                        let to_cell = fr * cols + fc;
                        if blocked[to_cell] { continue; }

                        let to_vel = classify_velocity(fvx, fvy);
                        let to_state = state_index(to_cell, to_vel) as u32;

                        // Only add one edge per resulting state (deduplicate)
                        if best_to_state != Some(to_state) {
                            // Check if we already recorded this target from another rotation
                            if !forward[from_state].iter().any(|e| e.to_state == to_state) {
                                best_to_state = Some(to_state);
                            }
                        }
                    }

                    if let Some(to) = best_to_state {
                        let edge = Edge { to_state: to, cost: SIM_FRAMES_PER_EDGE as u16 };
                        forward[from_state].push(edge);
                        reverse[to as usize].push(Edge { to_state: from_state as u32, cost: SIM_FRAMES_PER_EDGE as u16 });
                    }
                }
            }
        }

        (forward, reverse)
    }

    fn compute_time_to_reach(
        rows: usize, cols: usize, min_x: f32, min_y: f32,
        pickup_coords: &[(f32, f32)],
        reverse_edges: &[Vec<Edge>],
        num_states: usize,
    ) -> Vec<Vec<u16>> {
        let total_pickups = pickup_coords.len();
        let mut all_ttr = Vec::with_capacity(total_pickups);
        let collection_sq = PICKUP_COLLECTION_RADIUS * PICKUP_COLLECTION_RADIUS;

        for pickup_idx in 0..total_pickups {
            let (px, py) = pickup_coords[pickup_idx];
            let mut dist = vec![u16::MAX; num_states];
            let mut heap: BinaryHeap<Reverse<(u16, u32)>> = BinaryHeap::new();

            // Seed: all states within collection radius of pickup
            for r in 0..rows {
                for c in 0..cols {
                    let cx = min_x + (c as f32 + 0.5) * CELL_SIZE;
                    let cy = min_y + (r as f32 + 0.5) * CELL_SIZE;
                    let dx = cx - px;
                    let dy = cy - py;
                    if dx * dx + dy * dy <= collection_sq {
                        let cell_idx = r * cols + c;
                        // All velocity classes at this cell are goal states
                        for vel_class in 0..NUM_VEL_CLASSES {
                            let si = state_index(cell_idx, vel_class);
                            dist[si] = 0;
                            heap.push(Reverse((0, si as u32)));
                        }
                    }
                }
            }

            // Backward Dijkstra: expand from goal states using reverse edges
            while let Some(Reverse((d, state_u32))) = heap.pop() {
                let state = state_u32 as usize;
                if d > dist[state] { continue; }

                for edge in &reverse_edges[state] {
                    let from = edge.to_state as usize; // reverse edge: to_state is actually the predecessor
                    let nd = d.saturating_add(edge.cost);
                    if nd < dist[from] {
                        dist[from] = nd;
                        heap.push(Reverse((nd, from as u32)));
                    }
                }
            }

            all_ttr.push(dist);
        }

        all_ttr
    }

    fn compute_pickup_transit(
        rows: usize, cols: usize, min_x: f32, min_y: f32,
        pickup_coords: &[(f32, f32)],
        time_to_reach: &[Vec<u16>],
        total_pickups: usize,
    ) -> (Vec<Vec<Vec<u16>>>, Vec<Vec<Vec<u8>>>) {
        // pickup_transit[from][vel_class][to] = cost in frames
        let mut transit = vec![vec![vec![u16::MAX; total_pickups]; NUM_VEL_CLASSES]; total_pickups];
        let mut arrival = vec![vec![vec![0u8; total_pickups]; NUM_VEL_CLASSES]; total_pickups];

        for from in 0..total_pickups {
            let (fx, fy) = pickup_coords[from];
            let fc = ((fx - min_x) / CELL_SIZE) as i32;
            let fr = ((fy - min_y) / CELL_SIZE) as i32;
            if fr < 0 || fr >= rows as i32 || fc < 0 || fc >= cols as i32 { continue; }
            let from_cell = fr as usize * cols + fc as usize;

            for vel_class in 0..NUM_VEL_CLASSES {
                let si = state_index(from_cell, vel_class);

                for to in 0..total_pickups {
                    if to == from { continue; }
                    let cost = time_to_reach[to][si];
                    transit[from][vel_class][to] = cost;
                    // For arrival velocity, we'd need to trace the path — approximate as zero
                    arrival[from][vel_class][to] = 0;
                }
            }
        }

        (transit, arrival)
    }

    // --- Runtime query ---

    fn to_grid(&self, x: f32, y: f32) -> (usize, usize) {
        let c = ((x - self.min_x) / CELL_SIZE) as i32;
        let r = ((y - self.min_y) / CELL_SIZE) as i32;
        let r = r.clamp(0, self.rows as i32 - 1) as usize;
        let c = c.clamp(0, self.cols as i32 - 1) as usize;
        (r, c)
    }

    fn to_grid_unblocked(&self, x: f32, y: f32) -> (usize, usize) {
        let (r, c) = self.to_grid(x, y);
        if !self.blocked[r * self.cols + c] {
            return (r, c);
        }
        // Search nearby for unblocked cell
        for radius in 1..10 {
            for dr in -radius..=radius {
                for dc in -radius..=radius {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nr < self.rows as i32 && nc >= 0 && nc < self.cols as i32 {
                        let nr = nr as usize;
                        let nc = nc as usize;
                        if !self.blocked[nr * self.cols + nc] {
                            return (nr, nc);
                        }
                    }
                }
            }
        }
        (r, c)
    }

    /// Main query: returns (cost_as_distance, dir_x, dir_y)
    /// cost is in "equivalent distance" units for MCTS compatibility (frames * CELL_SIZE / SIM_FRAMES)
    pub fn get_nearest_pickup_info(
        &self,
        ship_x: f32, ship_y: f32,
        ship_vx: f32, ship_vy: f32,
        collected: &[bool],
    ) -> (f64, f64, f64) {
        let uncollected: Vec<usize> = (0..self.total_pickups).filter(|&i| !collected[i]).collect();
        if uncollected.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let (sr, sc) = self.to_grid_unblocked(ship_x, ship_y);
        let cell_idx = sr * self.cols + sc;
        let vel_class = classify_velocity(ship_vx, ship_vy);
        let si = state_index(cell_idx, vel_class);

        // Find best first target using momentum-aware greedy TSP
        let mut best_total: u32 = u32::MAX;
        let mut best_idx = uncollected[0];

        for &i in &uncollected {
            let cost_to_i = self.time_to_reach[i][si] as u32;
            if cost_to_i >= u16::MAX as u32 { continue; }

            // Greedy remaining cost using pickup transit matrix
            let remaining: Vec<usize> = uncollected.iter().copied().filter(|&j| j != i).collect();
            let remaining_cost = self.greedy_remaining_cost_momentum(i, vel_class, &remaining);

            let total = cost_to_i + remaining_cost;
            if total < best_total {
                best_total = total;
                best_idx = i;
            }
        }

        let time_frames = self.time_to_reach[best_idx][si];
        if time_frames == u16::MAX {
            // Unreachable — fall back to Euclidean
            let (tx, ty) = self.pickup_coords[best_idx];
            let dx = (tx - ship_x) as f64;
            let dy = (ty - ship_y) as f64;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > 0.0 {
                return (dist, dx / dist, dy / dist);
            }
            return (0.0, 0.0, 0.0);
        }

        // Convert frames to a distance-like cost for MCTS compatibility
        // Scale so that the magnitude is similar to the spatial pathfinder's output
        let cost_distance = time_frames as f64 * (CELL_SIZE as f64 / SIM_FRAMES_PER_EDGE as f64);

        // --- Direction computation ---
        let (tx, ty) = self.pickup_coords[best_idx];
        let euclid_dx = (tx - ship_x) as f64;
        let euclid_dy = (ty - ship_y) as f64;
        let euclid_dist = (euclid_dx * euclid_dx + euclid_dy * euclid_dy).sqrt();

        // Close range: direct Euclidean
        if euclid_dist < 150.0 && euclid_dist > 0.0 {
            return (euclid_dist.min(cost_distance), euclid_dx / euclid_dist, euclid_dy / euclid_dist);
        }

        // Momentum-aware direction: recommend thrust direction that best reduces
        // time-to-reach, by looking at which neighboring states have lower cost
        let dir = self.compute_momentum_direction(si, best_idx, ship_x, ship_y, ship_vx, ship_vy);
        match dir {
            Some((dx, dy)) => (cost_distance, dx, dy),
            None => {
                // Fallback: direct Euclidean
                if euclid_dist > 0.0 {
                    (cost_distance, euclid_dx / euclid_dist, euclid_dy / euclid_dist)
                } else {
                    (cost_distance, 0.0, 0.0)
                }
            }
        }
    }

    /// Compute recommended direction by finding which forward-edge neighbor has
    /// the lowest time-to-reach for the target pickup.
    fn compute_momentum_direction(
        &self,
        current_state: usize,
        target_pickup: usize,
        ship_x: f32, ship_y: f32,
        ship_vx: f32, ship_vy: f32,
    ) -> Option<(f64, f64)> {
        let current_cost = self.time_to_reach[target_pickup][current_state];
        if current_cost == 0 || current_cost == u16::MAX {
            return None;
        }

        // Look at all forward edges from current state, find the one with lowest TTR
        let mut best_cost = current_cost;
        let mut best_target_state: Option<u32> = None;

        for edge in &self.forward_edges[current_state] {
            let neighbor_cost = self.time_to_reach[target_pickup][edge.to_state as usize];
            let total = neighbor_cost.saturating_add(edge.cost);
            if total < best_cost {
                best_cost = total;
                best_target_state = Some(edge.to_state);
            }
        }

        if let Some(target_state) = best_target_state {
            // Extract the cell position of the best neighbor
            let target_cell = target_state as usize / NUM_VEL_CLASSES;
            let tr = target_cell / self.cols;
            let tc = target_cell % self.cols;
            let target_x = self.min_x + (tc as f32 + 0.5) * CELL_SIZE;
            let target_y = self.min_y + (tr as f32 + 0.5) * CELL_SIZE;

            // Also factor in the velocity class of the target state
            let target_vel_class = target_state as usize % NUM_VEL_CLASSES;
            let (desired_vx, desired_vy) = vel_class_to_vector(target_vel_class);

            // Blend positional direction with velocity correction
            let pos_dx = (target_x - ship_x) as f64;
            let pos_dy = (target_y - ship_y) as f64;
            let vel_dx = (desired_vx - ship_vx) as f64;
            let vel_dy = (desired_vy - ship_vy) as f64;

            // Weight: position direction + velocity correction
            let dx = pos_dx + vel_dx * 0.3;
            let dy = pos_dy + vel_dy * 0.3;
            let mag = (dx * dx + dy * dy).sqrt();
            if mag > 0.0 {
                return Some((dx / mag, dy / mag));
            }
        }

        None
    }

    fn greedy_remaining_cost_momentum(&self, start_pickup: usize, start_vel_class: usize, remaining: &[usize]) -> u32 {
        let mut cost: u32 = 0;
        let mut current = start_pickup;
        let mut current_vel = start_vel_class;
        let mut left: Vec<usize> = remaining.to_vec();

        while !left.is_empty() {
            let mut best_d = u16::MAX;
            let mut best_j = left[0];
            for &j in &left {
                let d = self.pickup_transit[current][current_vel][j];
                if d < best_d {
                    best_d = d;
                    best_j = j;
                }
            }
            if best_d >= u16::MAX { break; }
            cost += best_d as u32;
            // Approximate arrival velocity as zero (conservative)
            current_vel = self.pickup_arrival_vel[current][current_vel][best_j] as usize;
            left.retain(|&x| x != best_j);
            current = best_j;
        }
        cost
    }

    // --- Debug methods (matching PathfinderGrid interface) ---

    pub fn get_debug_path(&self, ship_x: f32, ship_y: f32, ship_vx: f32, ship_vy: f32, collected: &[bool]) -> Vec<(f32, f32)> {
        // Trace forward edges toward the target pickup
        let uncollected: Vec<usize> = (0..self.total_pickups).filter(|&i| !collected[i]).collect();
        if uncollected.is_empty() { return vec![]; }

        let (sr, sc) = self.to_grid_unblocked(ship_x, ship_y);
        let cell_idx = sr * self.cols + sc;
        let vel_class = classify_velocity(ship_vx, ship_vy);
        let mut current_state = state_index(cell_idx, vel_class);

        // Find target (same logic as get_nearest_pickup_info)
        let mut best_total: u32 = u32::MAX;
        let mut best_idx = uncollected[0];
        for &i in &uncollected {
            let cost = self.time_to_reach[i][current_state] as u32;
            if cost < best_total {
                best_total = cost;
                best_idx = i;
            }
        }

        let mut path = Vec::new();
        let mut visited = std::collections::HashSet::new();

        for _ in 0..200 { // max path length
            if !visited.insert(current_state) { break; }

            let cell = current_state / NUM_VEL_CLASSES;
            let r = cell / self.cols;
            let c = cell % self.cols;
            let x = self.min_x + (c as f32 + 0.5) * CELL_SIZE;
            let y = self.min_y + (r as f32 + 0.5) * CELL_SIZE;
            path.push((x, y));

            let current_cost = self.time_to_reach[best_idx][current_state];
            if current_cost == 0 { break; }

            // Follow best forward edge
            let mut best_next: Option<usize> = None;
            let mut best_next_cost = current_cost;
            for edge in &self.forward_edges[current_state] {
                let nc = self.time_to_reach[best_idx][edge.to_state as usize];
                if nc < best_next_cost {
                    best_next_cost = nc;
                    best_next = Some(edge.to_state as usize);
                }
            }

            match best_next {
                Some(next) => current_state = next,
                None => break,
            }
        }

        path
    }

    pub fn get_debug_target_info(
        &self,
        ship_x: f32, ship_y: f32,
        ship_vx: f32, ship_vy: f32,
        collected: &[bool],
    ) -> (i32, f32, f32, f64, f64, f64, f64) {
        let uncollected: Vec<usize> = (0..self.total_pickups).filter(|&i| !collected[i]).collect();
        if uncollected.is_empty() {
            return (-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let (sr, sc) = self.to_grid_unblocked(ship_x, ship_y);
        let cell_idx = sr * self.cols + sc;
        let vel_class = classify_velocity(ship_vx, ship_vy);
        let si = state_index(cell_idx, vel_class);

        let mut best_total: u32 = u32::MAX;
        let mut best_idx = uncollected[0];
        for &i in &uncollected {
            let cost = self.time_to_reach[i][si] as u32;
            if cost >= u16::MAX as u32 { continue; }
            let remaining: Vec<usize> = uncollected.iter().copied().filter(|&j| j != i).collect();
            let total = cost + self.greedy_remaining_cost_momentum(i, vel_class, &remaining);
            if total < best_total {
                best_total = total;
                best_idx = i;
            }
        }

        let (tx, ty) = self.pickup_coords[best_idx];
        let time_frames = self.time_to_reach[best_idx][si];
        let cost_distance = if time_frames < u16::MAX {
            time_frames as f64 * (CELL_SIZE as f64 / SIM_FRAMES_PER_EDGE as f64)
        } else {
            -1.0
        };
        let euclidean = (((ship_x - tx) as f64).powi(2) + ((ship_y - ty) as f64).powi(2)).sqrt();

        let (_, dir_x, dir_y) = self.get_nearest_pickup_info(ship_x, ship_y, ship_vx, ship_vy, collected);

        (best_idx as i32, tx, ty, cost_distance, euclidean, dir_x, dir_y)
    }
}

// --- PathfinderKind: enum dispatch for optional momentum pathfinder ---

pub enum PathfinderKind {
    Spatial(PathfinderGrid),
    Momentum(MomentumPathfinder),
}

impl PathfinderKind {
    /// Returns (cost, dir_x, dir_y). Spatial variant ignores velocity.
    pub fn get_nearest_pickup_info(
        &self,
        ship_x: f32, ship_y: f32,
        ship_vx: f32, ship_vy: f32,
        collected: &[bool],
    ) -> (f64, f64, f64) {
        match self {
            PathfinderKind::Spatial(pf) => pf.get_nearest_pickup_info(ship_x, ship_y, collected),
            PathfinderKind::Momentum(mpf) => mpf.get_nearest_pickup_info(ship_x, ship_y, ship_vx, ship_vy, collected),
        }
    }

    pub fn get_debug_path(
        &self,
        ship_x: f32, ship_y: f32,
        ship_vx: f32, ship_vy: f32,
        collected: &[bool],
    ) -> Vec<(f32, f32)> {
        match self {
            PathfinderKind::Spatial(pf) => pf.get_debug_path(ship_x, ship_y, collected),
            PathfinderKind::Momentum(mpf) => mpf.get_debug_path(ship_x, ship_y, ship_vx, ship_vy, collected),
        }
    }

    pub fn get_debug_target_info(
        &self,
        ship_x: f32, ship_y: f32,
        ship_vx: f32, ship_vy: f32,
        collected: &[bool],
    ) -> (i32, f32, f32, f64, f64, f64, f64) {
        match self {
            PathfinderKind::Spatial(pf) => pf.get_debug_target_info(ship_x, ship_y, collected),
            PathfinderKind::Momentum(mpf) => mpf.get_debug_target_info(ship_x, ship_y, ship_vx, ship_vy, collected),
        }
    }

    /// Route tangent (look-ahead direction along the optimal path) — spatial
    /// backend only. Momentum falls back to `get_nearest_pickup_info`'s
    /// pickup-direction, which is worse but preserves the existing contract.
    pub fn get_route_tangent(
        &self,
        ship_x: f32, ship_y: f32,
        ship_vx: f32, ship_vy: f32,
        collected: &[bool],
        look_ahead_cells: usize,
    ) -> (f64, f64, bool) {
        match self {
            PathfinderKind::Spatial(pf) => pf.get_route_tangent(ship_x, ship_y, collected, look_ahead_cells),
            PathfinderKind::Momentum(_) => {
                let (d, dx, dy) = self.get_nearest_pickup_info(ship_x, ship_y, ship_vx, ship_vy, collected);
                (dx, dy, d > 0.0)
            }
        }
    }

    /// Greedy-TSP remaining route length in world px (spatial backend).
    /// Momentum backend falls back to nearest-pickup distance — it doesn't
    /// precompute a pickup×pickup distance matrix, so full-tour cost isn't
    /// directly available there.
    pub fn get_remaining_route_length(
        &self,
        ship_x: f32, ship_y: f32,
        ship_vx: f32, ship_vy: f32,
        collected: &[bool],
    ) -> f64 {
        match self {
            PathfinderKind::Spatial(pf) => pf.get_remaining_route_length(ship_x, ship_y, collected),
            PathfinderKind::Momentum(mpf) => {
                mpf.get_nearest_pickup_info(ship_x, ship_y, ship_vx, ship_vy, collected).0
            }
        }
    }

    pub fn total_pickups(&self) -> usize {
        match self {
            PathfinderKind::Spatial(pf) => pf.total_pickups,
            PathfinderKind::Momentum(mpf) => mpf.total_pickups,
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            PathfinderKind::Spatial(pf) => pf.rows(),
            PathfinderKind::Momentum(mpf) => mpf.rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            PathfinderKind::Spatial(pf) => pf.cols(),
            PathfinderKind::Momentum(mpf) => mpf.cols,
        }
    }
}
