use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod real_physics;
pub mod real_collision;
pub mod real_map_parser;
pub mod real_game;
pub mod pathfinder;
pub mod mcts;
pub mod nn_evaluator;
pub mod alphazero_mcts;

use real_game::{RealSpaceAceGame, GameSnapshot};
use real_map_parser::parse_map_json;
use pathfinder::{PathfinderGrid, MomentumPathfinder, PathfinderKind};
use mcts::{mcts_search, mcts_search_with_stats, get_heuristic_breakdown, MCTSParams};
use nn_evaluator::{NNEvaluator, build_alphazero_obs};
use alphazero_mcts::{alphazero_search, AlphaZeroParams};

// ---------------------------------------------------------------------------
// Shared observation / reward logic (used by both PyO3 and IPC paths)
// ---------------------------------------------------------------------------

pub fn build_observation(game: &RealSpaceAceGame) -> Vec<f32> {
    let state = game.get_state();
    let mut obs = Vec::with_capacity(20);

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
    // Computed here to avoid redundant sin/cos in Python.
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

#[pyclass]
struct PyMCTSEngine {
    sim_game: RealSpaceAceGame,
    pathfinder: PathfinderKind,
    max_steps: u32,
}

#[pymethods]
impl PyMCTSEngine {
    #[new]
    #[pyo3(signature = (level, max_steps, use_momentum_pathfinder=false))]
    fn new(level: usize, max_steps: u32, use_momentum_pathfinder: bool) -> PyResult<Self> {
        let mut sim_game = RealSpaceAceGame::new();
        sim_game.load_level(level)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load level {}: {:?}", level, e)
            ))?;
        let pathfinder = if use_momentum_pathfinder {
            PathfinderKind::Momentum(MomentumPathfinder::build(&sim_game))
        } else {
            PathfinderKind::Spatial(PathfinderGrid::build(&sim_game))
        };
        Ok(PyMCTSEngine { sim_game, pathfinder, max_steps })
    }

    #[pyo3(signature = (state, num_simulations, action_repeat, c_explore, gamma, shaping_weight=0.5))]
    fn search(&mut self, state: &PyGameState, num_simulations: u32,
              action_repeat: u32, c_explore: f64, gamma: f64, shaping_weight: f64) -> u8 {
        let params = MCTSParams {
            num_simulations, action_repeat, c_explore, gamma,
            max_steps: self.max_steps, shaping_weight,
        };
        mcts_search(
            &mut self.sim_game,
            &self.pathfinder,
            state.snapshot.clone(),
            state.step_count,
            &params,
        )
    }

    fn get_debug_path(&self, state: &PyGameState) -> Vec<(f32, f32)> {
        let collected: Vec<bool> = state.snapshot.pickups.iter().map(|p| p.collected).collect();
        let ship_x = state.snapshot.physics.x;
        let ship_y = state.snapshot.physics.y;
        let ship_vx = state.snapshot.physics.vx;
        let ship_vy = state.snapshot.physics.vy;
        self.pathfinder.get_debug_path(ship_x, ship_y, ship_vx, ship_vy, &collected)
    }

    /// Returns (best_action, [(action_idx, visits, mean_value)], root_heuristic)
    #[pyo3(signature = (state, num_simulations, action_repeat, c_explore, gamma, shaping_weight=0.5))]
    fn search_with_stats(&mut self, state: &PyGameState, num_simulations: u32,
                         action_repeat: u32, c_explore: f64, gamma: f64, shaping_weight: f64)
        -> (u8, Vec<(u8, u32, f64)>, f64)
    {
        let params = MCTSParams {
            num_simulations, action_repeat, c_explore, gamma,
            max_steps: self.max_steps, shaping_weight,
        };
        mcts_search_with_stats(
            &mut self.sim_game,
            &self.pathfinder,
            state.snapshot.clone(),
            state.step_count,
            &params,
        )
    }

    /// Returns (path_distance, dir_x, dir_y) from pathfinder for current state
    fn get_pathfinder_stats(&self, state: &PyGameState) -> (f64, f64, f64) {
        let collected: Vec<bool> = state.snapshot.pickups.iter().map(|p| p.collected).collect();
        let ship_x = state.snapshot.physics.x;
        let ship_y = state.snapshot.physics.y;
        let ship_vx = state.snapshot.physics.vx;
        let ship_vy = state.snapshot.physics.vy;
        self.pathfinder.get_nearest_pickup_info(
            ship_x, ship_y, ship_vx, ship_vy, &collected
        )
    }

    /// Returns (target_idx, target_x, target_y, path_dist, euclidean_dist, dir_x, dir_y)
    fn get_debug_target_info(&self, state: &PyGameState) -> (i32, f32, f32, f64, f64, f64, f64) {
        let collected: Vec<bool> = state.snapshot.pickups.iter().map(|p| p.collected).collect();
        let ship_x = state.snapshot.physics.x;
        let ship_y = state.snapshot.physics.y;
        let ship_vx = state.snapshot.physics.vx;
        let ship_vy = state.snapshot.physics.vy;
        self.pathfinder.get_debug_target_info(ship_x, ship_y, ship_vx, ship_vy, &collected)
    }

    /// Returns heuristic breakdown dict for debug display
    fn get_heuristic_breakdown<'py>(&mut self, py: Python<'py>, state: &PyGameState) -> PyResult<Bound<'py, PyDict>> {
        let bd = get_heuristic_breakdown(
            &mut self.sim_game,
            &self.pathfinder,
            state.snapshot.clone(),
        );
        let d = PyDict::new(py);
        d.set_item("total", bd.total)?;
        d.set_item("pickups_score", bd.pickups_score)?;
        d.set_item("proximity_score", bd.proximity_score)?;
        d.set_item("velocity_score", bd.velocity_score)?;
        d.set_item("orientation_score", bd.orientation_score)?;
        d.set_item("tti_penalty", bd.tti_penalty)?;
        d.set_item("path_dist", bd.path_dist)?;
        d.set_item("dir_x", bd.dir_x)?;
        d.set_item("dir_y", bd.dir_y)?;
        d.set_item("speed_toward", bd.speed_toward)?;
        d.set_item("alignment", bd.alignment)?;
        d.set_item("min_tti", bd.min_tti)?;
        Ok(d)
    }

    /// Play multiple games using regular MCTS with dynamic sim/action_repeat scaling.
    /// Matches the behavior of the Python MCTSAgent (scales sims near walls, action_repeat with speed).
    /// Returns list of (completed, crashed, steps) per game.
    #[pyo3(signature = (num_games, num_simulations, action_repeat=5, c_explore=1.41, gamma=0.99, shaping_weight=0.5, max_steps=3000))]
    fn play_games(&mut self, num_games: u32, num_simulations: u32,
                  action_repeat: u32, c_explore: f64, gamma: f64,
                  shaping_weight: f64, max_steps: u32)
        -> Vec<(bool, bool, u32)>
    {
        let mut results = Vec::new();

        for _ in 0..num_games {
            self.sim_game.reset();
            let mut step: u32 = 0;

            loop {
                let snapshot = self.sim_game.save_state();

                // Dynamic scaling matching Python MCTSAgent behavior
                let state = self.sim_game.get_state();
                let speed = ((state.ship_vx * state.ship_vx + state.ship_vy * state.ship_vy) as f64).sqrt();
                let wall_distances = self.sim_game.get_wall_distances();
                let min_wall_dist = wall_distances.iter().cloned().fold(f32::INFINITY, f32::min) as f64;

                // action_repeat scales with speed
                let dynamic_ar = action_repeat + (speed / 50.0) as u32;

                // num_simulations scales near walls and at high speed
                let mut dynamic_sims = num_simulations as f64;
                if min_wall_dist < 150.0 {
                    dynamic_sims *= 1.0 + (150.0 - min_wall_dist) / 150.0;
                }
                dynamic_sims *= 1.0 + speed / 300.0;

                let params = MCTSParams {
                    num_simulations: dynamic_sims as u32,
                    action_repeat: dynamic_ar,
                    c_explore, gamma, max_steps, shaping_weight,
                };

                let action_idx = mcts_search(
                    &mut self.sim_game, &self.pathfinder,
                    snapshot.clone(), step, &params,
                );

                // Restore state after MCTS (search leaves sim_game in arbitrary state)
                self.sim_game.load_state(snapshot);

                let action = mcts::ACTIONS[action_idx as usize];
                let mut terminated = false;
                for _ in 0..dynamic_ar {
                    self.sim_game.set_controls(action[0], action[1], action[2]);
                    self.sim_game.step(1.0 / 60.0);
                    step += 1;
                    terminated = self.sim_game.is_terminated();
                    if terminated || step >= max_steps { break; }
                }

                if terminated || step >= max_steps {
                    let completed = self.sim_game.is_level_completed();
                    let crashed = self.sim_game.is_ship_exploded();
                    results.push((completed, crashed, step));
                    break;
                }
            }
        }

        results
    }

    fn get_pathfinder_info(&self) -> String {
        let kind = match &self.pathfinder {
            PathfinderKind::Spatial(_) => "spatial",
            PathfinderKind::Momentum(_) => "momentum",
        };
        format!("{}x{} grid, {} pickups ({})",
                self.pathfinder.rows(), self.pathfinder.cols(), self.pathfinder.total_pickups(), kind)
    }
}

#[pyclass]
struct PyPathfinder {
    kind: PathfinderKind,
}

#[pymethods]
impl PyPathfinder {
    #[new]
    #[pyo3(signature = (level, backend = "grid"))]
    fn new(level: usize, backend: &str) -> PyResult<Self> {
        let mut game = RealSpaceAceGame::new();
        game.load_level(level)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load level {}: {:?}", level, e)
            ))?;
        let kind = match backend {
            "grid" => PathfinderKind::Spatial(PathfinderGrid::build(&game)),
            "momentum" => PathfinderKind::Momentum(MomentumPathfinder::build(&game)),
            other => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("unknown backend: {other}")
            )),
        };
        Ok(PyPathfinder { kind })
    }

    /// Construct a pathfinder from a flat JSON array (the format produced by
    /// generate_maps.py's serialize_map). Always uses the grid backend.
    #[staticmethod]
    fn from_map_json(map_json: &str) -> PyResult<Self> {
        let map_data = parse_map_json(map_json)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Failed to parse map JSON"
            ))?;
        let mut game = RealSpaceAceGame::new();
        game.load_from_map_data(map_data);
        let pathfinder = PathfinderGrid::build(&game);
        Ok(PyPathfinder { kind: PathfinderKind::Spatial(pathfinder) })
    }

    /// Returns the active backend name.
    fn backend(&self) -> &'static str {
        match &self.kind {
            PathfinderKind::Spatial(_) => "grid",
            PathfinderKind::Momentum(_) => "momentum",
        }
    }

    /// Returns (all_reachable, per_pickup_path_distances).
    /// Unreachable pickups get f64::INFINITY. Grid backend only.
    fn validate_reachability(&self, spawn_x: f32, spawn_y: f32) -> PyResult<(bool, Vec<f64>)> {
        match &self.kind {
            PathfinderKind::Spatial(pf) => Ok(pf.validate_reachability(spawn_x, spawn_y)),
            PathfinderKind::Momentum(_) => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "validate_reachability is only available on the grid backend",
            )),
        }
    }

    /// Returns (path_distance, dir_x, dir_y) from pathfinder.
    /// Velocity defaults to zero; grid ignores it, momentum uses it.
    #[pyo3(signature = (ship_x, ship_y, collected, ship_vx=0.0, ship_vy=0.0))]
    fn get_nearest_pickup_info(&self, ship_x: f32, ship_y: f32, collected: Vec<bool>, ship_vx: f32, ship_vy: f32) -> (f64, f64, f64) {
        self.kind.get_nearest_pickup_info(ship_x, ship_y, ship_vx, ship_vy, &collected)
    }

    /// Returns (target_idx, target_x, target_y, path_dist, euclidean_dist, dir_x, dir_y)
    #[pyo3(signature = (ship_x, ship_y, collected, ship_vx=0.0, ship_vy=0.0))]
    fn get_debug_target_info(&self, ship_x: f32, ship_y: f32, collected: Vec<bool>, ship_vx: f32, ship_vy: f32) -> (i32, f32, f32, f64, f64, f64, f64) {
        self.kind.get_debug_target_info(ship_x, ship_y, ship_vx, ship_vy, &collected)
    }

    /// Returns (path_distance, dir_x, dir_y) toward a specific pickup index.
    /// Only available on the grid backend.
    fn get_distance_to_specific_pickup(&self, ship_x: f32, ship_y: f32, pickup_idx: usize) -> PyResult<(f64, f64, f64)> {
        match &self.kind {
            PathfinderKind::Spatial(pf) => Ok(pf.get_distance_to_specific_pickup(ship_x, ship_y, pickup_idx)),
            PathfinderKind::Momentum(_) => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "get_distance_to_specific_pickup is only available on the grid backend",
            )),
        }
    }

    /// Returns optimal TSP ordering of uncollected pickups using Held-Karp exact solver.
    /// Only available on the grid backend.
    fn get_tsp_order(&self, ship_x: f32, ship_y: f32, collected: Vec<bool>) -> PyResult<Vec<usize>> {
        match &self.kind {
            PathfinderKind::Spatial(pf) => Ok(pf.held_karp_tsp(ship_x, ship_y, &collected)),
            PathfinderKind::Momentum(_) => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "get_tsp_order is only available on the grid backend",
            )),
        }
    }

    /// Returns the full grid path from ship to a specific pickup as list of (x, y) tuples.
    /// Only available on the grid backend.
    fn get_path_to_specific_pickup(&self, ship_x: f32, ship_y: f32, pickup_idx: usize) -> PyResult<Vec<(f32, f32)>> {
        match &self.kind {
            PathfinderKind::Spatial(pf) => Ok(pf.get_path_to_specific_pickup(ship_x, ship_y, pickup_idx)),
            PathfinderKind::Momentum(_) => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "get_path_to_specific_pickup is only available on the grid backend",
            )),
        }
    }

    /// Returns pickup coordinates as list of (x, y) tuples.
    /// Only available on the grid backend.
    fn get_pickup_coords(&self) -> PyResult<Vec<(f32, f32)>> {
        match &self.kind {
            PathfinderKind::Spatial(pf) => Ok(pf.get_pickup_coords().to_vec()),
            PathfinderKind::Momentum(_) => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "get_pickup_coords is only available on the grid backend",
            )),
        }
    }

    /// Analyze level difficulty. Returns a dict of raw metrics.
    /// `ship_x, ship_y` = spawn position. Only available on grid backend.
    fn analyze_level_difficulty<'py>(&self, py: Python<'py>, ship_x: f32, ship_y: f32,
                                      map_lines: Vec<(f32, f32, f32, f32)>) -> PyResult<Bound<'py, PyDict>> {
        match &self.kind {
            PathfinderKind::Spatial(pf) => {
                let m = pf.analyze_difficulty(ship_x, ship_y, &map_lines);
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
            PathfinderKind::Momentum(_) => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "analyze_level_difficulty is only available on the grid backend",
            )),
        }
    }

    fn get_info(&self) -> String {
        let backend = match &self.kind {
            PathfinderKind::Spatial(_) => "grid",
            PathfinderKind::Momentum(_) => "momentum",
        };
        format!("{}x{} {} pathfinder, {} pickups",
                self.kind.rows(), self.kind.cols(), backend, self.kind.total_pickups())
    }
}

fn normalize_heuristic(heuristic: f64) -> f32 {
    // tanh gives strong gradient near 0 (crash avoidance) while saturating smoothly for high values.
    // heuristic=-100 (crash) -> -0.46, 0 (neutral) -> 0.0, 200 (1 pickup) -> 0.76
    (heuristic / 200.0).tanh() as f32
}

#[pyclass]
struct PyAlphaZeroEngine {
    sim_game: RealSpaceAceGame,
    pathfinder: PathfinderKind,
    nn: Option<NNEvaluator>,
    max_steps: u32,
}

#[pymethods]
impl PyAlphaZeroEngine {
    #[new]
    #[pyo3(signature = (level, max_steps, model_path=None))]
    fn new(level: usize, max_steps: u32, model_path: Option<String>) -> PyResult<Self> {
        let mut sim_game = RealSpaceAceGame::new();
        sim_game.load_level(level)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load level {}: {:?}", level, e)
            ))?;
        let pathfinder = PathfinderKind::Spatial(PathfinderGrid::build(&sim_game));
        let nn = match model_path {
            Some(path) => Some(NNEvaluator::load(&path).map_err(|e|
                pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Failed to load ONNX model {}: {:?}", path, e)
                )
            )?),
            None => None,
        };
        Ok(PyAlphaZeroEngine { sim_game, pathfinder, nn, max_steps })
    }

    /// Run AlphaZero MCTS search.
    /// Returns (best_action, policy_target[6], root_value).
    #[pyo3(signature = (state, num_simulations, c_puct=1.5, temperature=1.0, action_repeat=5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25))]
    fn search(&mut self, state: &PyGameState, num_simulations: u32,
              c_puct: f64, temperature: f64, action_repeat: u32,
              dirichlet_alpha: f64, dirichlet_epsilon: f64)
        -> (u8, Vec<f32>, f32)
    {
        let params = AlphaZeroParams {
            num_simulations, action_repeat, c_puct,
            max_steps: self.max_steps, temperature,
            dirichlet_alpha, dirichlet_epsilon,
        };
        let (action, policy, value, _obs) = alphazero_search(
            &mut self.sim_game,
            &self.pathfinder,
            &mut self.nn,
            state.snapshot.clone(),
            state.step_count,
            &params,
        );
        (action, policy.to_vec(), value)
    }

    /// Load or replace the neural network model.
    fn load_model(&mut self, model_path: String) -> PyResult<()> {
        self.nn = Some(NNEvaluator::load(&model_path).map_err(|e|
            pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load ONNX model {}: {:?}", model_path, e)
            )
        )?);
        Ok(())
    }

    /// Build the 27-dim observation for a given state (for training data collection).
    fn get_observation(&mut self, state: &PyGameState) -> Vec<f32> {
        self.sim_game.load_state(state.snapshot.clone());
        build_alphazero_obs(&self.sim_game, &self.pathfinder, state.step_count, self.max_steps)
    }

    /// Evaluate a game state using the heuristic function.
    /// Returns value normalized to [-1, 1] for use as value target in training.
    fn evaluate_heuristic(&mut self, state: &PyGameState) -> f32 {
        self.sim_game.load_state(state.snapshot.clone());
        let heuristic = mcts::evaluate_state_pub(&self.sim_game, &self.pathfinder);
        normalize_heuristic(heuristic)
    }

    /// Play multiple self-play games entirely in Rust.
    /// Returns (observations_flat, policies_flat, values, stats_list).
    /// observations_flat is a flat Vec<f32> of obs_dim * N examples.
    /// policies_flat is a flat Vec<f32> of 6 * N examples.
    /// values is Vec<f32> of N examples.
    /// stats_list is a list of (total_reward, pickups_collected, completed, crashed, steps).
    #[pyo3(signature = (num_games, num_simulations, c_puct=1.5, action_repeat=5, temp_threshold=30, max_steps=3000, dirichlet_alpha=0.3, dirichlet_epsilon=0.25))]
    fn play_games(&mut self, num_games: u32, num_simulations: u32,
                  c_puct: f64, action_repeat: u32, temp_threshold: u32, max_steps: u32,
                  dirichlet_alpha: f64, dirichlet_epsilon: f64)
        -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<(f64, i32, bool, bool, u32)>)
    {
        let mut all_obs: Vec<f32> = Vec::new();
        let mut all_policies: Vec<f32> = Vec::new();
        let mut all_values: Vec<f32> = Vec::new();
        let mut all_stats: Vec<(f64, i32, bool, bool, u32)> = Vec::new();

        for _ in 0..num_games {
            self.sim_game.reset();
            let mut step: u32 = 0;
            let mut total_reward: f64 = 0.0;
            let mut pickups_collected: i32 = 0;

            // Per-game example buffers (obs/policy/value per decision step)
            let mut game_obs: Vec<f32> = Vec::new();
            let mut game_policies: Vec<f32> = Vec::new();
            let mut game_values: Vec<f32> = Vec::new();

            loop {
                let snapshot = self.sim_game.save_state();
                let temperature = if step < temp_threshold { 1.0 } else { 0.1 };

                let params = AlphaZeroParams {
                    num_simulations, action_repeat, c_puct,
                    max_steps, temperature,
                    dirichlet_alpha, dirichlet_epsilon,
                };

                let (action_idx, policy, _value, obs) = alphazero_search(
                    &mut self.sim_game,
                    &self.pathfinder,
                    &mut self.nn,
                    snapshot,
                    step,
                    &params,
                );

                // Store observation and policy
                game_obs.extend_from_slice(&obs);
                game_policies.extend_from_slice(&policy);

                // Execute action with action_repeat
                let action = alphazero_mcts::ACTIONS[action_idx as usize];
                let mut terminated = false;
                for _ in 0..action_repeat {
                    self.sim_game.set_controls(action[0], action[1], action[2]);
                    self.sim_game.step(1.0 / 60.0);
                    step += 1;

                    // Accumulate reward
                    total_reward += calculate_reward(&self.sim_game) as f64;
                    pickups_collected += self.sim_game.get_pickups_collected_this_step();

                    terminated = self.sim_game.is_terminated();
                    if terminated || step >= max_steps { break; }
                }

                // Placeholder — value assigned retroactively after game ends
                game_values.push(0.0);

                if terminated || step >= max_steps {
                    let completed = self.sim_game.is_level_completed();
                    let crashed = self.sim_game.is_ship_exploded();

                    // Assign discounted game outcome as value target
                    // Wins get a speed bonus: finishing faster = higher value
                    let time_remaining = 1.0 - (step as f64 / max_steps as f64);
                    let outcome: f64 = if completed {
                        // Range [0.5, 1.0]: fast completion = 1.0, slow = 0.5
                        0.5 + 0.5 * time_remaining
                    } else if crashed { -1.0 }
                    else { 0.0 }; // truncated
                    let n = game_values.len();
                    let discount: f64 = 0.99;
                    for i in 0..n {
                        let steps_from_end = (n - 1 - i) as f64;
                        game_values[i] = (outcome * discount.powf(steps_from_end)) as f32;
                    }

                    all_obs.extend_from_slice(&game_obs);
                    all_policies.extend_from_slice(&game_policies);
                    all_values.extend_from_slice(&game_values);
                    all_stats.push((total_reward, pickups_collected, completed, crashed, step));
                    break;
                }
            }
        }

        (all_obs, all_policies, all_values, all_stats)
    }

    fn get_pathfinder_info(&self) -> String {
        format!("{}x{} grid, {} pickups (alphazero)",
                self.pathfinder.rows(), self.pathfinder.cols(), self.pathfinder.total_pickups())
    }
}

#[pymodule]
fn spaceace_rl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameInstance>()?;
    m.add_class::<PyGameState>()?;
    m.add_class::<PyMCTSEngine>()?;
    m.add_class::<PyPathfinder>()?;
    m.add_class::<PyAlphaZeroEngine>()?;
    Ok(())
}
