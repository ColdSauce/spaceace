use std::cell::Cell;

use crate::pathfinder::PathfinderKind;
use crate::real_game::{GameSnapshot, RealSpaceAceGame};

// Simple fast RNG (xorshift32)
thread_local! {
    static RNG_STATE: Cell<u32> = Cell::new(12345);
}

fn fast_random() -> u32 {
    RNG_STATE.with(|state| {
        let mut x = state.get();
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state.set(x);
        x
    })
}

// 6 useful actions: [rotate_left, rotate_right, thrust]
pub const ACTIONS: [[bool; 3]; 6] = [
    [false, false, false], // coast
    [false, false, true],  // thrust
    [true, false, false],  // rotate left
    [true, false, true],   // rotate left + thrust
    [false, true, false],  // rotate right
    [false, true, true],   // rotate right + thrust
];
const NUM_ACTIONS: usize = 6;

// 8 ship-relative directions for wall distance raycasts
const BASE_DIRS: [(f64, f64); 8] = [
    (0.0, -1.0), (0.707, -0.707), (1.0, 0.0), (0.707, 0.707),
    (0.0, 1.0), (-0.707, 0.707), (-1.0, 0.0), (-0.707, -0.707),
];

pub struct MCTSParams {
    pub num_simulations: u32,
    pub action_repeat: u32,
    pub c_explore: f64,
    pub gamma: f64,
    pub max_steps: u32,
    pub shaping_weight: f64,
}

struct MCTSNode {
    snapshot: GameSnapshot,
    step_count: u32,
    action: i8,        // index into ACTIONS, -1 for root
    parent: i32,       // index into nodes vec, -1 for root
    children: Vec<usize>,
    visit_count: u32,
    total_reward: f64,
    is_terminal: bool,
    untried_actions: Vec<u8>,
}

impl MCTSNode {
    fn new(snapshot: GameSnapshot, step_count: u32, action: i8, parent: i32) -> Self {
        let mut untried: Vec<u8> = (0..NUM_ACTIONS as u8).collect();
        for i in (1..untried.len()).rev() {
            let j = fast_random() as usize % (i + 1);
            untried.swap(i, j);
        }
        MCTSNode {
            snapshot, step_count, action, parent,
            children: Vec::new(),
            visit_count: 0,
            total_reward: 0.0,
            is_terminal: false,
            untried_actions: untried,
        }
    }
}

/// Heuristic breakdown for debug display
pub struct HeuristicBreakdown {
    pub total: f64,
    pub pickups_score: f64,
    pub proximity_score: f64,
    pub velocity_score: f64,
    pub orientation_score: f64,
    pub tangential_penalty: f64,
    pub tti_penalty: f64,
    pub path_dist: f64,
    pub dir_x: f64,
    pub dir_y: f64,
    pub speed_toward: f64,
    pub alignment: f64,
    pub min_tti: f64,
}

fn evaluate_state_breakdown(game: &RealSpaceAceGame, pathfinder: &PathfinderKind) -> HeuristicBreakdown {
    let mut bd = HeuristicBreakdown {
        total: 0.0, pickups_score: 0.0, proximity_score: 0.0,
        velocity_score: 0.0, orientation_score: 0.0, tangential_penalty: 0.0,
        tti_penalty: 0.0, path_dist: 0.0, dir_x: 0.0, dir_y: 0.0,
        speed_toward: 0.0, alignment: 0.0, min_tti: f64::INFINITY,
    };

    if game.is_level_completed() {
        bd.total = pathfinder.total_pickups() as f64 * 200.0 + 1000.0;
        return bd;
    }
    if game.is_ship_exploded() { bd.total = -1000.0; return bd; }

    let state = game.get_state();
    let ship_x = state.ship_x;
    let ship_y = state.ship_y;
    let ship_vx = state.ship_vx;
    let ship_vy = state.ship_vy;
    let ship_rot = state.ship_rotation as f64;
    let pickups_remaining = state.pickups_remaining;

    let collected: Vec<bool> = game.get_pickup_positions().iter().map(|&(_, _, c)| c).collect();
    let (path_dist, dir_x, dir_y) = pathfinder.get_nearest_pickup_info(ship_x, ship_y, ship_vx, ship_vy, &collected);
    bd.path_dist = path_dist;
    bd.dir_x = dir_x;
    bd.dir_y = dir_y;

    // Pickups collected
    bd.pickups_score = (pathfinder.total_pickups() as f64 - pickups_remaining as f64) * 200.0;

    // When path_dist is small, switch to euclidean distance to the pickup.
    // The pathfinder grid can't route into the wall inflation zone, but the
    // ship CAN fly there to collect. Use euclidean once we're within ~60px
    // so the heuristic has gradient all the way to collection.
    let (guide_dist, guide_dx, guide_dy) = if path_dist > 0.0 && path_dist < 80.0 {
        // Get target pickup position and compute direct euclidean vector
        let collected: Vec<bool> = game.get_pickup_positions().iter().map(|&(_, _, c)| c).collect();
        let target_info = pathfinder.get_debug_target_info(ship_x, ship_y, ship_vx, ship_vy, &collected);
        let tx = target_info.1 as f64;
        let ty = target_info.2 as f64;
        let edx = tx - ship_x as f64;
        let edy = ty - ship_y as f64;
        let edist = (edx * edx + edy * edy).sqrt();
        if edist > 1.0 {
            (edist, edx / edist, edy / edist)
        } else {
            (edist, dir_x, dir_y)
        }
    } else {
        (path_dist, dir_x, dir_y)
    };

    // Path proximity: exponential bonus near the pickup creates gradient toward it.
    // Bonus only (never negative) — being far from the *next* pickup shouldn't
    // penalize collecting the *current* one.
    if guide_dist > 0.0 {
        bd.proximity_score = 100.0 * (-guide_dist / 50.0).exp();
    }

    // Velocity toward pickup
    if guide_dist > 0.0 {
        bd.speed_toward = ship_vx as f64 * guide_dx + ship_vy as f64 * guide_dy;
        bd.velocity_score = bd.speed_toward * 0.3;
    }

    // Ship orientation alignment
    if guide_dist > 0.0 {
        let heading_x = ship_rot.sin();
        let heading_y = -ship_rot.cos();
        bd.alignment = heading_x * guide_dx + heading_y * guide_dy;
        let orient_weight = (guide_dist / 200.0).min(1.0) * 20.0;
        bd.orientation_score = bd.alignment * orient_weight;
    }

    // TTI wall avoidance
    let wall_distances = game.get_wall_distances();
    let cos_r = ship_rot.cos();
    let sin_r = ship_rot.sin();
    let speed = ((ship_vx * ship_vx + ship_vy * ship_vy) as f64).sqrt();
    let mut min_wall_dist = f64::INFINITY;

    for (i, &(dx, dy)) in BASE_DIRS.iter().enumerate() {
        let world_dx = dx * cos_r - dy * sin_r;
        let world_dy = dx * sin_r + dy * cos_r;
        let v_toward = ship_vx as f64 * world_dx + ship_vy as f64 * world_dy;
        let wall_dist = wall_distances[i] as f64;
        if wall_dist < min_wall_dist { min_wall_dist = wall_dist; }
        if v_toward > 1.0 {
            let tti = wall_dist / v_toward;
            if tti < bd.min_tti { bd.min_tti = tti; }
        }
    }

    // Scale TTI threshold with speed: faster ships need to react earlier
    let tti_threshold = 0.4 + (speed / 200.0).min(0.6);
    if bd.min_tti < tti_threshold {
        // Quadratic penalty — much steeper as TTI approaches zero.
        // At tti_threshold=1.0, tti=0: penalty = -1.0^2 * 500 = -500
        // At tti=0.3 below threshold: penalty = -0.3^2 * 500 = -45
        let deficit = tti_threshold - bd.min_tti;
        bd.tti_penalty = -deficit * deficit * 500.0;
        // Hard floor: if TTI < 0.15s, this is near-certain death
        if bd.min_tti < 0.15 {
            bd.tti_penalty -= 500.0;
        }
    }

    // Direct wall proximity penalty: regardless of velocity direction,
    // being very close to walls is dangerous (gravity, rotation can
    // quickly change velocity direction into the wall).
    if min_wall_dist < 40.0 {
        let prox_penalty = (40.0 - min_wall_dist) / 40.0; // 0..1
        bd.tti_penalty -= prox_penalty * prox_penalty * 200.0 * (1.0 + speed / 100.0);
    }

    bd.total = bd.pickups_score + bd.proximity_score + bd.velocity_score + bd.orientation_score + bd.tti_penalty;
    bd
}

fn evaluate_state(game: &RealSpaceAceGame, pathfinder: &PathfinderKind) -> f64 {
    evaluate_state_breakdown(game, pathfinder).total
}

/// Public wrapper for AlphaZero fallback evaluation
pub fn evaluate_state_pub(game: &RealSpaceAceGame, pathfinder: &PathfinderKind) -> f64 {
    evaluate_state(game, pathfinder)
}

pub fn get_heuristic_breakdown(
    game: &mut RealSpaceAceGame,
    pathfinder: &PathfinderKind,
    snapshot: GameSnapshot,
) -> HeuristicBreakdown {
    game.load_state(snapshot);
    evaluate_state_breakdown(game, pathfinder)
}

fn uct_score(child: &MCTSNode, c_explore: f64, parent_log_visits: f64) -> f64 {
    if child.visit_count == 0 {
        return f64::INFINITY;
    }
    let exploitation = child.total_reward / child.visit_count as f64;
    let exploration = c_explore * (parent_log_visits / child.visit_count as f64).sqrt();
    exploitation + exploration
}

fn best_uct_child(nodes: &[MCTSNode], node_idx: usize, c_explore: f64) -> usize {
    let parent_log = (nodes[node_idx].visit_count as f64).ln();
    let mut best_idx = nodes[node_idx].children[0];
    let mut best_score = f64::NEG_INFINITY;
    for &child_idx in &nodes[node_idx].children {
        let score = uct_score(&nodes[child_idx], c_explore, parent_log);
        if score > best_score {
            best_score = score;
            best_idx = child_idx;
        }
    }
    best_idx
}

pub fn mcts_search(
    game: &mut RealSpaceAceGame,
    pathfinder: &PathfinderKind,
    root_snapshot: GameSnapshot,
    root_step_count: u32,
    params: &MCTSParams,
) -> u8 {
    let mut nodes: Vec<MCTSNode> = Vec::with_capacity(params.num_simulations as usize + 16);
    nodes.push(MCTSNode::new(root_snapshot.clone(), root_step_count, -1, -1));

    // Compute root baseline so backpropagated values are relative improvements
    game.load_state(root_snapshot);
    let root_baseline = evaluate_state(game, pathfinder);

    for _ in 0..params.num_simulations {
        let mut node_idx: usize = 0;

        // --- SELECTION ---
        while nodes[node_idx].untried_actions.is_empty()
            && !nodes[node_idx].children.is_empty()
            && !nodes[node_idx].is_terminal
        {
            node_idx = best_uct_child(&nodes, node_idx, params.c_explore);
        }

        // --- EXPANSION ---
        let mut reward: f64 = 0.0;
        if !nodes[node_idx].untried_actions.is_empty() && !nodes[node_idx].is_terminal {
            let action_idx = nodes[node_idx].untried_actions.pop().unwrap();
            let action = ACTIONS[action_idx as usize];

            game.load_state(nodes[node_idx].snapshot.clone());

            // Reward shaping: measure path distance before action
            let collected_before: Vec<bool> = game.get_pickup_positions().iter().map(|&(_, _, c)| c).collect();
            let state_before = game.get_state();
            let (dist_before, _, _) = pathfinder.get_nearest_pickup_info(
                state_before.ship_x, state_before.ship_y,
                state_before.ship_vx, state_before.ship_vy, &collected_before);

            let mut step_count = nodes[node_idx].step_count;
            let mut terminated = false;
            let mut truncated = false;

            for _ in 0..params.action_repeat {
                game.set_controls(action[0], action[1], action[2]);
                game.step(1.0 / 60.0);
                step_count += 1;
                reward += crate::calculate_reward(game) as f64;
                terminated = game.is_terminated();
                truncated = step_count >= params.max_steps;
                if terminated || truncated { break; }
            }

            // Reward shaping: measure path distance after action
            // Skip when a pickup was collected — dist_after would be to a different
            // (farther) pickup, creating a huge false penalty
            if !terminated {
                let collected_after: Vec<bool> = game.get_pickup_positions().iter().map(|&(_, _, c)| c).collect();
                let pickup_collected = collected_before.iter().zip(collected_after.iter()).any(|(b, a)| b != a);
                if !pickup_collected {
                    let state_after = game.get_state();
                    let (dist_after, _, _) = pathfinder.get_nearest_pickup_info(
                        state_after.ship_x, state_after.ship_y,
                        state_after.ship_vx, state_after.ship_vy, &collected_after);
                    reward += params.shaping_weight * (dist_before - dist_after);
                }
            }

            let child_snapshot = game.save_state();
            let child_idx = nodes.len();
            let mut child = MCTSNode::new(child_snapshot, step_count, action_idx as i8, node_idx as i32);
            child.is_terminal = terminated || truncated;
            nodes.push(child);
            nodes[node_idx].children.push(child_idx);
            node_idx = child_idx;
        }

        // --- EVALUATION (relative to root baseline) ---
        game.load_state(nodes[node_idx].snapshot.clone());
        let heuristic = evaluate_state(game, pathfinder);
        let value = reward + params.gamma * (heuristic - root_baseline);

        // --- BACKPROPAGATION ---
        let mut idx = node_idx as i32;
        while idx >= 0 {
            let i = idx as usize;
            nodes[i].visit_count += 1;
            nodes[i].total_reward += value;
            idx = nodes[i].parent;
        }
    }

    // Return action of most-visited child of root
    if nodes[0].children.is_empty() {
        return 1; // fallback: thrust
    }
    let mut best_child = nodes[0].children[0];
    let mut best_visits = 0u32;
    for &child_idx in &nodes[0].children {
        if nodes[child_idx].visit_count > best_visits {
            best_visits = nodes[child_idx].visit_count;
            best_child = child_idx;
        }
    }
    nodes[best_child].action as u8
}

/// Like mcts_search but also returns per-action stats: (best_action, [(action_idx, visits, mean_value)], root_heuristic)
pub fn mcts_search_with_stats(
    game: &mut RealSpaceAceGame,
    pathfinder: &PathfinderKind,
    root_snapshot: GameSnapshot,
    root_step_count: u32,
    params: &MCTSParams,
) -> (u8, Vec<(u8, u32, f64)>, f64) {
    let mut nodes: Vec<MCTSNode> = Vec::with_capacity(params.num_simulations as usize + 16);
    nodes.push(MCTSNode::new(root_snapshot.clone(), root_step_count, -1, -1));

    // Compute root baseline so backpropagated values are relative improvements
    game.load_state(root_snapshot.clone());
    let root_baseline = evaluate_state(game, pathfinder);

    for _ in 0..params.num_simulations {
        let mut node_idx: usize = 0;

        while nodes[node_idx].untried_actions.is_empty()
            && !nodes[node_idx].children.is_empty()
            && !nodes[node_idx].is_terminal
        {
            node_idx = best_uct_child(&nodes, node_idx, params.c_explore);
        }

        let mut reward: f64 = 0.0;
        if !nodes[node_idx].untried_actions.is_empty() && !nodes[node_idx].is_terminal {
            let action_idx = nodes[node_idx].untried_actions.pop().unwrap();
            let action = ACTIONS[action_idx as usize];

            game.load_state(nodes[node_idx].snapshot.clone());

            // Reward shaping: measure path distance before action
            let collected_before: Vec<bool> = game.get_pickup_positions().iter().map(|&(_, _, c)| c).collect();
            let state_before = game.get_state();
            let (dist_before, _, _) = pathfinder.get_nearest_pickup_info(
                state_before.ship_x, state_before.ship_y,
                state_before.ship_vx, state_before.ship_vy, &collected_before);

            let mut step_count = nodes[node_idx].step_count;
            let mut terminated = false;
            let mut truncated = false;

            for _ in 0..params.action_repeat {
                game.set_controls(action[0], action[1], action[2]);
                game.step(1.0 / 60.0);
                step_count += 1;
                reward += crate::calculate_reward(game) as f64;
                terminated = game.is_terminated();
                truncated = step_count >= params.max_steps;
                if terminated || truncated { break; }
            }

            // Reward shaping: measure path distance after action
            // Skip when a pickup was collected — dist_after would be to a different
            // (farther) pickup, creating a huge false penalty
            if !terminated {
                let collected_after: Vec<bool> = game.get_pickup_positions().iter().map(|&(_, _, c)| c).collect();
                let pickup_collected = collected_before.iter().zip(collected_after.iter()).any(|(b, a)| b != a);
                if !pickup_collected {
                    let state_after = game.get_state();
                    let (dist_after, _, _) = pathfinder.get_nearest_pickup_info(
                        state_after.ship_x, state_after.ship_y,
                        state_after.ship_vx, state_after.ship_vy, &collected_after);
                    reward += params.shaping_weight * (dist_before - dist_after);
                }
            }

            let child_snapshot = game.save_state();
            let child_idx = nodes.len();
            let mut child = MCTSNode::new(child_snapshot, step_count, action_idx as i8, node_idx as i32);
            child.is_terminal = terminated || truncated;
            nodes.push(child);
            nodes[node_idx].children.push(child_idx);
            node_idx = child_idx;
        }

        // --- EVALUATION (relative to root baseline) ---
        game.load_state(nodes[node_idx].snapshot.clone());
        let heuristic = evaluate_state(game, pathfinder);
        let value = reward + params.gamma * (heuristic - root_baseline);

        let mut idx = node_idx as i32;
        while idx >= 0 {
            let i = idx as usize;
            nodes[i].visit_count += 1;
            nodes[i].total_reward += value;
            idx = nodes[i].parent;
        }
    }

    // Collect per-action stats from root children
    let mut action_stats: Vec<(u8, u32, f64)> = Vec::new();
    for &child_idx in &nodes[0].children {
        let child = &nodes[child_idx];
        let mean_val = if child.visit_count > 0 {
            child.total_reward / child.visit_count as f64
        } else {
            0.0
        };
        action_stats.push((child.action as u8, child.visit_count, mean_val));
    }

    let root_heuristic = root_baseline;

    let best_action = if nodes[0].children.is_empty() {
        1
    } else {
        let mut best_child = nodes[0].children[0];
        let mut best_visits = 0u32;
        for &child_idx in &nodes[0].children {
            if nodes[child_idx].visit_count > best_visits {
                best_visits = nodes[child_idx].visit_count;
                best_child = child_idx;
            }
        }
        nodes[best_child].action as u8
    };

    (best_action, action_stats, root_heuristic)
}
