use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod real_physics;
pub mod real_collision;
pub mod real_map_parser;
pub mod real_game;
pub mod pathfinder;
pub mod solver;

use real_game::{RealSpaceAceGame, GameSnapshot};
use real_map_parser::parse_map_json;
use pathfinder::PathfinderGrid;
use solver::{AceSolver, BeamParams};

// ---------------------------------------------------------------------------
// Shared observation / reward logic
// ---------------------------------------------------------------------------

pub fn build_observation(game: &RealSpaceAceGame) -> Vec<f32> {
    let state = game.get_state();
    let mut obs = Vec::with_capacity(36);

    // Ship state (5 values)
    obs.push(state.ship_x);
    obs.push(state.ship_y);
    obs.push(state.ship_vx);
    obs.push(state.ship_vy);
    obs.push(state.ship_rotation);

    // Closest pickup (3 values)
    let (pickup_x, pickup_y, pickup_dist) = game.get_closest_pickup();
    obs.push(pickup_x);
    obs.push(pickup_y);
    obs.push(pickup_dist);

    // Wall distances in 8 directions (8 values) — ship-relative
    let wall_distances = game.get_wall_distances();
    obs.extend_from_slice(&wall_distances);

    // Pickups remaining (1 value)
    obs.push(state.pickups_remaining as f32);

    // Normalized position within map bounds (2 values)
    let bounds = game.get_map_bounds();
    let w = bounds.max_x - bounds.min_x;
    let h = bounds.max_y - bounds.min_y;
    obs.push(if w > 0.0 { (state.ship_x - bounds.min_x) / w } else { 0.5 });
    obs.push(if h > 0.0 { (state.ship_y - bounds.min_y) / h } else { 0.5 });

    // Minimum time-to-impact across 8 raycast directions (1 value, index 19)
    let cos_r = state.ship_rotation.cos();
    let sin_r = state.ship_rotation.sin();
    let base_dirs: [(f32, f32); 8] = [
        (0.0, -1.0), (0.707, -0.707), (1.0, 0.0), (0.707, 0.707),
        (0.0, 1.0), (-0.707, 0.707), (-1.0, 0.0), (-0.707, -0.707),
    ];
    let mut min_tti: f32 = f32::INFINITY;
    for (i, &(dx, dy)) in base_dirs.iter().enumerate() {
        let world_dx = dx * cos_r - dy * sin_r;
        let world_dy = dx * sin_r + dy * cos_r;
        let v_toward = state.ship_vx * world_dx + state.ship_vy * world_dy;
        if v_toward > 1.0 {
            let tti = wall_distances[i] / v_toward;
            if tti < min_tti {
                min_tti = tti;
            }
        }
    }
    obs.push(min_tti);

    // Fine wall distances: 16 extra rays (indices 20..36), interleaved with the
    // existing 8 to form 24 evenly spaced rays at 15° in ship-local space.
    let fine = game.get_wall_distances_fine_16();
    obs.extend_from_slice(&fine);

    obs
}

pub fn calculate_reward(game: &RealSpaceAceGame) -> f32 {
    let mut reward: f32 = -0.01;

    if game.is_ship_exploded() {
        reward -= 100.0;
    }
    if game.is_level_completed() {
        reward += 1000.0;
    }

    reward += game.get_pickups_collected_this_step() as f32 * 50.0;

    let (_, _, closest_pickup_dist) = game.get_closest_pickup();
    if closest_pickup_dist < 100.0 {
        reward += (100.0 - closest_pickup_dist) * 0.01;
    }

    reward
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
struct PyGameState {
    snapshot: GameSnapshot,
    step_count: u32,
}

#[pyclass]
struct PyGameInstance {
    game: RealSpaceAceGame,
    current_level: usize,
    step_count: u32,
    max_steps: u32,
}

#[pymethods]
impl PyGameInstance {
    #[new]
    fn new(level: usize, max_steps: u32) -> PyResult<Self> {
        let mut game = RealSpaceAceGame::new();
        game.load_level(level)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load level {}: {:?}", level, e)
            ))?;

        Ok(PyGameInstance {
            game,
            current_level: level,
            step_count: 0,
            max_steps,
        })
    }

    fn reset(&mut self) -> Vec<f32> {
        self.game.reset();
        self.step_count = 0;
        build_observation(&self.game)
    }

    fn step<'py>(&mut self, py: Python<'py>, action: [i32; 3]) -> PyResult<(Vec<f32>, f32, bool, bool, Bound<'py, PyDict>)> {
        self.game.set_controls(
            action[0] > 0,
            action[1] > 0,
            action[2] > 0,
        );
        self.game.step(1.0 / 60.0);
        self.step_count += 1;

        let observation = build_observation(&self.game);
        let reward = calculate_reward(&self.game);
        let terminated = self.game.is_terminated();
        let truncated = self.step_count >= self.max_steps;

        let info = PyDict::new(py);
        info.set_item("step_count", self.step_count)?;
        info.set_item("pickups_remaining", self.game.get_pickups_remaining())?;
        info.set_item("ship_exploded", self.game.is_ship_exploded())?;
        info.set_item("level_completed", self.game.is_level_completed())?;

        Ok((observation, reward, terminated, truncated, info))
    }

    fn get_observation(&self) -> Vec<f32> {
        build_observation(&self.game)
    }

    fn get_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = PyDict::new(py);
        let state = self.game.get_state();
        info.set_item("level", self.current_level)?;
        info.set_item("step_count", self.step_count)?;
        info.set_item("max_steps", self.max_steps)?;
        info.set_item("pickups_remaining", self.game.get_pickups_remaining())?;
        info.set_item("ship_exploded", self.game.is_ship_exploded())?;
        info.set_item("level_completed", self.game.is_level_completed())?;
        info.set_item("map_lines_count", self.game.get_map_lines().len())?;

        let pos = PyDict::new(py);
        pos.set_item("x", state.ship_x)?;
        pos.set_item("y", state.ship_y)?;
        info.set_item("ship_position", pos)?;

        Ok(info)
    }

    fn get_level_info(&self) -> String {
        format!("Level {} loaded with {} map lines",
                self.current_level,
                self.game.get_map_lines().len())
    }

    fn get_map_geometry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);

        // Map lines as list of [x1, y1, x2, y2]
        let map_lines: Vec<Vec<f32>> = self.game.get_map_lines()
            .iter()
            .map(|&(x1, y1, x2, y2)| vec![x1, y1, x2, y2])
            .collect();
        result.set_item("map_lines", map_lines)?;

        // Pickup positions as list of (x, y, collected)
        let pickup_positions: Vec<(f32, f32, bool)> = self.game.get_pickup_positions();
        result.set_item("pickup_positions", pickup_positions)?;

        // Bounds
        let bounds_dict = PyDict::new(py);
        let bounds = self.game.get_map_bounds();
        bounds_dict.set_item("min_x", bounds.min_x)?;
        bounds_dict.set_item("max_x", bounds.max_x)?;
        bounds_dict.set_item("min_y", bounds.min_y)?;
        bounds_dict.set_item("max_y", bounds.max_y)?;
        result.set_item("bounds", bounds_dict)?;

        Ok(result)
    }

    fn get_pickup_states(&self) -> Vec<bool> {
        self.game.get_pickup_positions().iter().map(|&(_, _, collected)| collected).collect()
    }

    fn save_state(&self) -> PyGameState {
        PyGameState {
            snapshot: self.game.save_state(),
            step_count: self.step_count,
        }
    }

    fn load_state(&mut self, state: &PyGameState) {
        self.game.load_state(state.snapshot.clone());
        self.step_count = state.step_count;
    }

    fn render_ascii(&self) -> String {
        self.game.render_ascii()
    }

    fn render_detailed(&self) -> String {
        self.game.render_detailed()
    }
}

/// Grid pathfinder for level tooling (reachability validation, difficulty
/// analysis, route visualization). Not used by the solver.
#[pyclass]
struct PyPathfinder {
    grid: PathfinderGrid,
}

#[pymethods]
impl PyPathfinder {
    #[new]
    #[pyo3(signature = (level, backend = "grid"))]
    fn new(level: usize, backend: &str) -> PyResult<Self> {
        if backend != "grid" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("unknown backend: {backend} (only \"grid\" is supported)")
            ));
        }
        let mut game = RealSpaceAceGame::new();
        game.load_level(level)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load level {}: {:?}", level, e)
            ))?;
        Ok(PyPathfinder { grid: PathfinderGrid::build(&game) })
    }

    /// Construct a pathfinder from a flat JSON array (the format produced by
    /// generate_maps.py's serialize_map).
    #[staticmethod]
    fn from_map_json(map_json: &str) -> PyResult<Self> {
        let map_data = parse_map_json(map_json)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Failed to parse map JSON"
            ))?;
        let mut game = RealSpaceAceGame::new();
        game.load_from_map_data(map_data);
        Ok(PyPathfinder { grid: PathfinderGrid::build(&game) })
    }

    fn backend(&self) -> &'static str {
        "grid"
    }

    /// Returns (all_reachable, per_pickup_path_distances).
    /// Unreachable pickups get f64::INFINITY.
    fn validate_reachability(&self, spawn_x: f32, spawn_y: f32) -> (bool, Vec<f64>) {
        self.grid.validate_reachability(spawn_x, spawn_y)
    }

    /// Returns (path_distance, dir_x, dir_y) toward the nearest uncollected pickup.
    #[pyo3(signature = (ship_x, ship_y, collected, ship_vx=0.0, ship_vy=0.0))]
    fn get_nearest_pickup_info(&self, ship_x: f32, ship_y: f32, collected: Vec<bool>, ship_vx: f32, ship_vy: f32) -> (f64, f64, f64) {
        let _ = (ship_vx, ship_vy);
        self.grid.get_nearest_pickup_info(ship_x, ship_y, &collected)
    }

    /// Returns (target_idx, target_x, target_y, path_dist, euclidean_dist, dir_x, dir_y)
    #[pyo3(signature = (ship_x, ship_y, collected, ship_vx=0.0, ship_vy=0.0))]
    fn get_debug_target_info(&self, ship_x: f32, ship_y: f32, collected: Vec<bool>, ship_vx: f32, ship_vy: f32) -> (i32, f32, f32, f64, f64, f64, f64) {
        let _ = (ship_vx, ship_vy);
        self.grid.get_debug_target_info(ship_x, ship_y, &collected)
    }

    /// Returns (path_distance, dir_x, dir_y) toward a specific pickup index.
    fn get_distance_to_specific_pickup(&self, ship_x: f32, ship_y: f32, pickup_idx: usize) -> (f64, f64, f64) {
        self.grid.get_distance_to_specific_pickup(ship_x, ship_y, pickup_idx)
    }

    /// Returns optimal TSP ordering of uncollected pickups (Held-Karp).
    fn get_tsp_order(&self, ship_x: f32, ship_y: f32, collected: Vec<bool>) -> Vec<usize> {
        self.grid.held_karp_tsp(ship_x, ship_y, &collected)
    }

    /// Returns the full grid path from ship to a specific pickup as (x, y) tuples.
    fn get_path_to_specific_pickup(&self, ship_x: f32, ship_y: f32, pickup_idx: usize) -> Vec<(f32, f32)> {
        self.grid.get_path_to_specific_pickup(ship_x, ship_y, pickup_idx)
    }

    /// Returns pickup coordinates as (x, y) tuples.
    fn get_pickup_coords(&self) -> Vec<(f32, f32)> {
        self.grid.get_pickup_coords().to_vec()
    }

    /// Analyze level difficulty. Returns a dict of raw metrics.
    fn analyze_level_difficulty<'py>(&self, py: Python<'py>, ship_x: f32, ship_y: f32,
                                      map_lines: Vec<(f32, f32, f32, f32)>) -> PyResult<Bound<'py, PyDict>> {
        let m = self.grid.analyze_difficulty(ship_x, ship_y, &map_lines);
        let d = PyDict::new(py);
        d.set_item("num_walls", m.num_walls)?;
        d.set_item("wall_density", m.wall_density)?;
        d.set_item("num_pickups", m.num_pickups)?;
        d.set_item("pickup_spread", m.pickup_spread)?;
        d.set_item("total_route_length", m.total_route_length)?;
        d.set_item("detour_ratio", m.detour_ratio)?;
        d.set_item("bottleneck_clearance", m.bottleneck_clearance)?;
        d.set_item("upward_travel", m.upward_travel)?;
        d.set_item("upward_travel_tight", m.upward_travel_tight)?;
        d.set_item("maneuver_count", m.maneuver_count)?;
        d.set_item("worst_maneuver_angle", m.worst_maneuver_angle)?;
        d.set_item("map_width", m.map_width)?;
        d.set_item("map_height", m.map_height)?;
        Ok(d)
    }

    fn get_info(&self) -> String {
        format!("{}x{} grid pathfinder, {} pickups",
                self.grid.rows(), self.grid.cols(), self.grid.total_pickups)
    }
}

/// Offline time-optimal planner (see src/solver.rs). This is the AI:
/// `solve` finds a completing action tape via parallel beam search, `refine`
/// improves it inside a corridor around the incumbent, `polish` shortens it
/// with exact local search, `resolve_suffix` re-plans tails. All returned
/// tapes replay tick-exactly on PyGameInstance.
#[pyclass]
struct PySolver {
    inner: AceSolver,
}

#[pymethods]
impl PySolver {
    /// `strict=True` (default) checks wall collision every tick, so planned
    /// tapes never overlap a wall. `strict=False` models the engine exactly,
    /// including its every-other-frame collision skip at speed — tapes may
    /// legally thread walls on skipped frames.
    #[new]
    #[pyo3(signature = (level, strict=true))]
    fn new(level: usize, strict: bool) -> PyResult<Self> {
        let inner = AceSolver::from_level(level, strict)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load level {}: {:?}", level, e)
            ))?;
        Ok(PySolver { inner })
    }

    fn n_pickups(&self) -> usize {
        self.inner.n_pickups()
    }

    /// Beam-search a completing tape from spawn. Returns None if the beam
    /// dies or max_ticks is exhausted.
    #[pyo3(signature = (width=50_000, max_ticks=4000, seed=0, quant_pos=6.0, quant_vel=12.0, rot_bins=64, lookahead=1.0, mix=0.8, proj_div=700.0, doom_scale=1.0, turn_w=1.0, jitter=3.0))]
    #[allow(clippy::too_many_arguments)]
    fn solve(&self, py: Python<'_>, width: usize, max_ticks: u32, seed: u64,
             quant_pos: f32, quant_vel: f32, rot_bins: u32,
             lookahead: f32, mix: f32, proj_div: f32, doom_scale: f32, turn_w: f32, jitter: f32) -> Option<Vec<u8>> {
        let p = BeamParams { width, max_ticks, seed, quant_pos, quant_vel, rot_bins, lookahead, mix, proj_div, doom_scale, turn_w, jitter };
        py.allow_threads(|| self.inner.solve(&p))
    }

    /// Exact replay. Returns (completed, crashed, ticks).
    fn replay(&self, py: Python<'_>, tape: Vec<u8>) -> (bool, bool, u32) {
        py.allow_threads(|| self.inner.replay(&tape))
    }

    /// Heuristic probe: remaining-route lower bound (px) at a position.
    fn h_at(&self, x: f32, y: f32, mask: u32) -> i64 {
        self.inner.h_at(x, y, mask)
    }

    /// Trace of (tick, x, y, h) along a tape, sampled every `stride` ticks.
    fn trace(&self, tape: Vec<u8>, stride: usize) -> Vec<(u32, f32, f32, i64)> {
        self.inner.trace(&tape, stride)
    }

    /// Local-search polish: `chains` parallel chains of `iters` mutations
    /// each; returns (best_tape, best_ticks).
    #[pyo3(signature = (tape, iters=300_000, chains=8, seed=1, accept_equal=0.12))]
    fn polish(&self, py: Python<'_>, tape: Vec<u8>, iters: u64, chains: usize,
              seed: u64, accept_equal: f64) -> (Vec<u8>, u32) {
        py.allow_threads(|| self.inner.polish(&tape, iters, chains, seed, accept_equal))
    }

    /// Corridor refinement: re-search inside a `radius`-px tube around the
    /// reference tape with (typically finer) quantization. Returns a strictly
    /// shorter tape or None.
    #[pyo3(signature = (tape, radius=150.0, width=100_000, seed=0, quant_pos=3.0, quant_vel=6.0, rot_bins=96, lookahead=1.0, mix=1.0, proj_div=400.0, doom_scale=0.1, turn_w=1.0, jitter=3.0))]
    #[allow(clippy::too_many_arguments)]
    fn refine(&self, py: Python<'_>, tape: Vec<u8>, radius: f32, width: usize,
              seed: u64, quant_pos: f32, quant_vel: f32, rot_bins: u32,
              lookahead: f32, mix: f32, proj_div: f32, doom_scale: f32, turn_w: f32, jitter: f32) -> Option<Vec<u8>> {
        let p = BeamParams { width, max_ticks: 0, seed, quant_pos, quant_vel, rot_bins, lookahead, mix, proj_div, doom_scale, turn_w, jitter };
        py.allow_threads(|| self.inner.refine(&tape, radius, &p))
    }

    /// Rendezvous prefix re-solve: find a shorter path from spawn to the
    /// tape's state at `rendezvous_tick` (within tolerances), splice the
    /// original suffix, and validate by exact replay. Returns a strictly
    /// shorter tape or None.
    #[pyo3(signature = (tape, rendezvous_tick, tol_pos=3.0, tol_vel=5.0, tol_rot=0.04, width=100_000, seed=0, quant_pos=3.0, quant_vel=6.0, rot_bins=96, lookahead=1.0, mix=1.0, proj_div=300.0, doom_scale=0.3, turn_w=1.0, jitter=3.0))]
    #[allow(clippy::too_many_arguments)]
    fn resolve_prefix(&self, py: Python<'_>, tape: Vec<u8>, rendezvous_tick: usize,
                      tol_pos: f32, tol_vel: f32, tol_rot: f32,
                      width: usize, seed: u64, quant_pos: f32, quant_vel: f32,
                      rot_bins: u32, lookahead: f32, mix: f32, proj_div: f32,
                      doom_scale: f32, turn_w: f32, jitter: f32) -> Option<Vec<u8>> {
        let p = BeamParams { width, max_ticks: 0, seed, quant_pos, quant_vel, rot_bins, lookahead, mix, proj_div, doom_scale, turn_w, jitter };
        py.allow_threads(|| self.inner.resolve_prefix(&tape, rendezvous_tick, tol_pos, tol_vel, tol_rot, &p))
    }

    /// Re-plan the suffix from `from_tick`; returns a strictly shorter full
    /// tape or None.
    #[pyo3(signature = (tape, from_tick, width=50_000, seed=0, quant_pos=6.0, quant_vel=12.0, rot_bins=64, lookahead=1.0, mix=0.8, proj_div=700.0, doom_scale=1.0, turn_w=1.0, jitter=3.0))]
    #[allow(clippy::too_many_arguments)]
    fn resolve_suffix(&self, py: Python<'_>, tape: Vec<u8>, from_tick: usize,
                      width: usize, seed: u64, quant_pos: f32, quant_vel: f32,
                      rot_bins: u32, lookahead: f32, mix: f32, proj_div: f32, doom_scale: f32, turn_w: f32, jitter: f32) -> Option<Vec<u8>> {
        let p = BeamParams { width, max_ticks: 0, seed, quant_pos, quant_vel, rot_bins, lookahead, mix, proj_div, doom_scale, turn_w, jitter };
        py.allow_threads(|| self.inner.resolve_suffix(&tape, from_tick, &p))
    }
}

#[pymodule]
fn spaceace_rl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameInstance>()?;
    m.add_class::<PyGameState>()?;
    m.add_class::<PyPathfinder>()?;
    m.add_class::<PySolver>()?;
    Ok(())
}
