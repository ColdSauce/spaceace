use ort::session::Session;
use ort::value::Tensor;

use crate::build_observation;
use crate::pathfinder::PathfinderKind;
use crate::real_game::RealSpaceAceGame;

// 8 ship-relative directions for wall distance raycasts (matches mcts.rs BASE_DIRS)
const BASE_DIRS: [(f64, f64); 8] = [
    (0.0, -1.0), (0.707, -0.707), (1.0, 0.0), (0.707, 0.707),
    (0.0, 1.0), (-0.707, 0.707), (-1.0, 0.0), (-0.707, -0.707),
];

/// Build the 27-dim observation used by AlphaZero.
/// All features normalized to approximately [-1, 1] or [0, 1] range.
/// First 19 dims from game state (normalized), then 8 pathfinder-derived features.
pub fn build_alphazero_obs(
    game: &RealSpaceAceGame,
    pathfinder: &PathfinderKind,
    step_count: u32,
    max_steps: u32,
) -> Vec<f32> {
    let state = game.get_state();
    let bounds = game.get_map_bounds();
    let map_w = (bounds.max_x - bounds.min_x).max(1.0);
    let map_h = (bounds.max_y - bounds.min_y).max(1.0);
    let map_diag = (map_w * map_w + map_h * map_h).sqrt();
    const MAX_SPEED: f32 = 300.0;
    const MAX_WALL_DIST: f32 = 500.0;

    let mut obs = Vec::with_capacity(27);

    // Ship state (5 values) — normalized
    obs.push((state.ship_x - bounds.min_x) / map_w);           // [0, 1]
    obs.push((state.ship_y - bounds.min_y) / map_h);           // [0, 1]
    obs.push(state.ship_vx / MAX_SPEED);                       // ~[-1, 1]
    obs.push(state.ship_vy / MAX_SPEED);                       // ~[-1, 1]
    obs.push(state.ship_rotation / std::f32::consts::PI);      // [-1, 1]

    // Closest pickup (3 values) — normalized
    let (pickup_x, pickup_y, pickup_dist) = game.get_closest_pickup();
    obs.push((pickup_x - bounds.min_x) / map_w);               // [0, 1]
    obs.push((pickup_y - bounds.min_y) / map_h);               // [0, 1]
    obs.push((pickup_dist / map_diag).min(1.0));                // [0, 1]

    // Wall distances in 8 directions (8 values) — normalized
    let wall_distances = game.get_wall_distances();
    for &wd in &wall_distances {
        obs.push((wd / MAX_WALL_DIST).min(1.0));               // [0, 1]
    }

    // Pickups remaining (1 value) — normalized by total
    let total_pickups = pathfinder.total_pickups().max(1) as f32;
    obs.push(state.pickups_remaining as f32 / total_pickups);   // [0, 1]

    // Normalized position within map bounds (2 values) — already normalized
    obs.push((state.ship_x - bounds.min_x) / map_w);
    obs.push((state.ship_y - bounds.min_y) / map_h);
    let ship_x = state.ship_x;
    let ship_y = state.ship_y;
    let ship_vx = state.ship_vx;
    let ship_vy = state.ship_vy;
    let ship_rot = state.ship_rotation;

    let collected: Vec<bool> = game.get_pickup_positions().iter().map(|&(_, _, c)| c).collect();
    let (path_dist, dir_x, dir_y) = pathfinder.get_nearest_pickup_info(
        ship_x, ship_y, ship_vx, ship_vy, &collected,
    );

    // 1. Normalized path distance
    let path_dist_norm = (path_dist as f32 / 1000.0).min(1.0);

    // 2-3. Direction to target (already normalized)
    let dir_x = dir_x as f32;
    let dir_y = dir_y as f32;

    // 4. Speed
    let speed = ((ship_vx * ship_vx + ship_vy * ship_vy) as f64).sqrt();

    // 5. Velocity toward target (normalized by speed)
    let speed_toward = if speed > 1e-6 && (dir_x.abs() > 1e-6 || dir_y.abs() > 1e-6) {
        (ship_vx as f64 * dir_x as f64 + ship_vy as f64 * dir_y as f64) / speed
    } else {
        0.0
    };

    // 6. Heading alignment
    let heading_x = (ship_rot as f64).sin();
    let heading_y = -(ship_rot as f64).cos();
    let heading_alignment = heading_x * dir_x as f64 + heading_y * dir_y as f64;

    // 7. Min TTI
    let wall_distances = game.get_wall_distances();
    let cos_r = (ship_rot as f64).cos();
    let sin_r = (ship_rot as f64).sin();
    let mut min_tti: f64 = f64::INFINITY;
    for (i, &(dx, dy)) in BASE_DIRS.iter().enumerate() {
        let world_dx = dx * cos_r - dy * sin_r;
        let world_dy = dx * sin_r + dy * cos_r;
        let v_toward = ship_vx as f64 * world_dx + ship_vy as f64 * world_dy;
        if v_toward > 1.0 {
            let tti = wall_distances[i] as f64 / v_toward;
            if tti < min_tti { min_tti = tti; }
        }
    }
    let min_tti_norm = min_tti.min(2.0) / 2.0;

    // 8. Time remaining
    let time_remaining = 1.0 - step_count as f32 / max_steps as f32;

    obs.push(path_dist_norm);
    obs.push(dir_x);
    obs.push(dir_y);
    obs.push(speed as f32);
    obs.push(speed_toward as f32);
    obs.push(heading_alignment as f32);
    obs.push(min_tti_norm as f32);
    obs.push(time_remaining);

    obs
}

pub struct NNEvaluator {
    session: Session,
}

impl NNEvaluator {
    pub fn load(model_path: &str) -> Result<Self, ort::Error> {
        let session = Session::builder()?
            .with_intra_threads(2)?
            .commit_from_file(model_path)?;
        Ok(NNEvaluator { session })
    }

    /// Run inference. Returns (policy_priors[6], value).
    /// Policy priors are softmax probabilities.
    pub fn evaluate(&mut self, obs: &[f32]) -> (Vec<f32>, f32) {
        let input = Tensor::from_array(([1, obs.len()], obs.to_vec().into_boxed_slice()))
            .expect("failed to create input tensor");

        let outputs = self.session.run(ort::inputs![input])
            .expect("ONNX inference failed");

        // Output 0: policy logits [1, 6]
        let (_, policy_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("failed to extract policy tensor");
        let policy_logits: Vec<f32> = policy_data.iter().copied().collect();

        // Softmax
        let max_logit = policy_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = policy_logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let policy: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

        // Output 1: value [1, 1]
        let (_, value_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .expect("failed to extract value tensor");
        let value = value_data.iter().next().copied().unwrap_or(0.0);

        (policy, value)
    }
}
