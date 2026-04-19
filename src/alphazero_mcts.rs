use std::cell::Cell;

use crate::pathfinder::PathfinderKind;
use crate::mcts::evaluate_state_pub;
use crate::nn_evaluator::{build_alphazero_obs, NNEvaluator};
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

fn fast_random_f64() -> f64 {
    (fast_random() as f64) / (u32::MAX as f64)
}

/// Sample from Gamma(alpha, 1) using Marsaglia and Tsang's method.
/// For alpha < 1, uses the boost: Gamma(a) = Gamma(a+1) * U^(1/a).
fn sample_gamma(alpha: f64) -> f64 {
    if alpha < 1.0 {
        let u = fast_random_f64().max(1e-10);
        return sample_gamma(alpha + 1.0) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Box-Muller for normal sample
        let u1 = fast_random_f64().max(1e-10);
        let u2 = fast_random_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 { continue; }
        let u = fast_random_f64().max(1e-10);
        if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// Sample a Dirichlet(alpha, ..., alpha) vector of given length.
fn sample_dirichlet(alpha: f64, n: usize) -> Vec<f64> {
    let mut samples: Vec<f64> = (0..n).map(|_| sample_gamma(alpha).max(1e-10)).collect();
    let total: f64 = samples.iter().sum();
    for s in &mut samples {
        *s /= total;
    }
    samples
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

pub struct AlphaZeroParams {
    pub num_simulations: u32,
    pub action_repeat: u32,
    pub c_puct: f64,
    pub max_steps: u32,
    pub temperature: f64,
    pub dirichlet_alpha: f64,
    pub dirichlet_epsilon: f64,
}

struct AZNode {
    snapshot: GameSnapshot,
    step_count: u32,
    action: i8,
    parent: i32,
    children: Vec<usize>,
    visit_count: u32,
    total_value: f64,
    prior: f64,
    is_terminal: bool,
    is_expanded: bool,
}

impl AZNode {
    fn new(snapshot: GameSnapshot, step_count: u32, action: i8, parent: i32, prior: f64) -> Self {
        AZNode {
            snapshot, step_count, action, parent,
            children: Vec::new(),
            visit_count: 0,
            total_value: 0.0,
            prior,
            is_terminal: false,
            is_expanded: false,
        }
    }

    fn q_value(&self) -> f64 {
        if self.visit_count == 0 { 0.0 }
        else { self.total_value / self.visit_count as f64 }
    }
}

fn puct_score(child: &AZNode, c_puct: f64, parent_visits_sqrt: f64) -> f64 {
    child.q_value() + c_puct * child.prior * parent_visits_sqrt / (1.0 + child.visit_count as f64)
}

fn best_puct_child(nodes: &[AZNode], node_idx: usize, c_puct: f64) -> usize {
    let parent_visits_sqrt = (nodes[node_idx].visit_count as f64).sqrt();
    let mut best_idx = nodes[node_idx].children[0];
    let mut best_score = f64::NEG_INFINITY;
    for &child_idx in &nodes[node_idx].children {
        let score = puct_score(&nodes[child_idx], c_puct, parent_visits_sqrt);
        if score > best_score {
            best_score = score;
            best_idx = child_idx;
        }
    }
    best_idx
}

/// Evaluate a leaf node. If nn is Some, use the neural network.
/// Otherwise fall back to uniform prior + heuristic value.
fn evaluate_leaf(
    game: &mut RealSpaceAceGame,
    pathfinder: &PathfinderKind,
    nn: &mut Option<NNEvaluator>,
    snapshot: &GameSnapshot,
    step_count: u32,
    max_steps: u32,
) -> ([f64; NUM_ACTIONS], f64) {
    game.load_state(snapshot.clone());

    if game.is_level_completed() {
        return ([1.0 / NUM_ACTIONS as f64; NUM_ACTIONS], 1.0);
    }
    if game.is_ship_exploded() {
        return ([1.0 / NUM_ACTIONS as f64; NUM_ACTIONS], -1.0);
    }

    match nn.as_mut() {
        Some(evaluator) => {
            let obs = build_alphazero_obs(game, pathfinder, step_count, max_steps);
            let (policy_vec, value) = evaluator.evaluate(&obs);
            let mut policy = [0.0f64; NUM_ACTIONS];
            for i in 0..NUM_ACTIONS.min(policy_vec.len()) {
                policy[i] = policy_vec[i] as f64;
            }
            (policy, value as f64)
        }
        None => {
            // Fallback: uniform prior, heuristic value normalized via tanh (preserves
            // gradient across the full heuristic range; /200 matches normalize_heuristic
            // used in training value targets).
            let heuristic = evaluate_state_pub(game, pathfinder);
            let value = (heuristic / 200.0).tanh();
            ([1.0 / NUM_ACTIONS as f64; NUM_ACTIONS], value)
        }
    }
}

/// Expand a node: simulate all actions and create children.
fn expand_node(
    nodes: &mut Vec<AZNode>,
    node_idx: usize,
    game: &mut RealSpaceAceGame,
    policy: &[f64; NUM_ACTIONS],
    params: &AlphaZeroParams,
) {
    let parent_snapshot = nodes[node_idx].snapshot.clone();
    let parent_step_count = nodes[node_idx].step_count;

    for action_idx in 0..NUM_ACTIONS {
        let action = ACTIONS[action_idx];
        game.load_state(parent_snapshot.clone());

        let mut step_count = parent_step_count;
        let mut terminated = false;
        for _ in 0..params.action_repeat {
            game.set_controls(action[0], action[1], action[2]);
            game.step(1.0 / 60.0);
            step_count += 1;
            terminated = game.is_terminated();
            if terminated || step_count >= params.max_steps { break; }
        }

        let child_snapshot = game.save_state();
        let child_idx = nodes.len();
        let mut child = AZNode::new(
            child_snapshot, step_count, action_idx as i8, node_idx as i32, policy[action_idx],
        );
        child.is_terminal = terminated || step_count >= params.max_steps;
        nodes.push(child);
        nodes[node_idx].children.push(child_idx);
    }
    nodes[node_idx].is_expanded = true;
}

/// Run AlphaZero MCTS search.
/// Returns (best_action, policy_target[6], root_value, root_observation).
/// nn is borrowed mutably — pass None for heuristic fallback.
pub fn alphazero_search(
    game: &mut RealSpaceAceGame,
    pathfinder: &PathfinderKind,
    nn: &mut Option<NNEvaluator>,
    root_snapshot: GameSnapshot,
    root_step_count: u32,
    params: &AlphaZeroParams,
) -> (u8, [f32; NUM_ACTIONS], f32, Vec<f32>) {
    let mut nodes: Vec<AZNode> = Vec::with_capacity(params.num_simulations as usize + 16);

    // Create root
    nodes.push(AZNode::new(root_snapshot.clone(), root_step_count, -1, -1, 1.0));

    // Build root observation (cached for return to caller)
    game.load_state(root_snapshot.clone());
    let root_obs = build_alphazero_obs(game, pathfinder, root_step_count, params.max_steps);

    // Evaluate root to get policy priors for children
    let (mut root_policy, _) = evaluate_leaf(
        game, pathfinder, nn, &root_snapshot, root_step_count, params.max_steps,
    );

    // Add Dirichlet noise at root for exploration
    if params.dirichlet_epsilon > 0.0 {
        let noise = sample_dirichlet(params.dirichlet_alpha, NUM_ACTIONS);
        let eps = params.dirichlet_epsilon;
        for i in 0..NUM_ACTIONS {
            root_policy[i] = (1.0 - eps) * root_policy[i] + eps * noise[i];
        }
    }

    // Expand root
    expand_node(&mut nodes, 0, game, &root_policy, params);

    for _ in 0..params.num_simulations {
        let mut node_idx: usize = 0;

        // --- SELECTION ---
        while nodes[node_idx].is_expanded && !nodes[node_idx].children.is_empty() && !nodes[node_idx].is_terminal {
            node_idx = best_puct_child(&nodes, node_idx, params.c_puct);
        }

        // --- EXPANSION + EVALUATION ---
        let value = if nodes[node_idx].is_terminal {
            game.load_state(nodes[node_idx].snapshot.clone());
            if game.is_level_completed() { 1.0 }
            else if game.is_ship_exploded() { -1.0 }
            else { 0.0 }
        } else if !nodes[node_idx].is_expanded {
            let (policy, leaf_value) = evaluate_leaf(
                game, pathfinder, nn,
                &nodes[node_idx].snapshot,
                nodes[node_idx].step_count,
                params.max_steps,
            );
            expand_node(&mut nodes, node_idx, game, &policy, params);
            leaf_value
        } else {
            0.0
        };

        // --- BACKPROPAGATION ---
        let mut idx = node_idx as i32;
        while idx >= 0 {
            let i = idx as usize;
            nodes[i].visit_count += 1;
            nodes[i].total_value += value;
            idx = nodes[i].parent;
        }
    }

    // --- ACTION SELECTION ---
    let mut visit_counts = [0u32; NUM_ACTIONS];
    let mut total_visits = 0u32;
    for &child_idx in &nodes[0].children {
        let action = nodes[child_idx].action as usize;
        visit_counts[action] = nodes[child_idx].visit_count;
        total_visits += nodes[child_idx].visit_count;
    }

    // Policy target: normalized visit counts
    let mut policy_target = [0.0f32; NUM_ACTIONS];
    if total_visits > 0 {
        for i in 0..NUM_ACTIONS {
            policy_target[i] = visit_counts[i] as f32 / total_visits as f32;
        }
    }

    // Select action based on temperature
    let best_action = if params.temperature < 0.01 {
        let mut best = 0usize;
        let mut best_visits = 0u32;
        for i in 0..NUM_ACTIONS {
            if visit_counts[i] > best_visits {
                best_visits = visit_counts[i];
                best = i;
            }
        }
        best as u8
    } else {
        let inv_temp = 1.0 / params.temperature;
        let mut weights = [0.0f64; NUM_ACTIONS];
        for i in 0..NUM_ACTIONS {
            weights[i] = (visit_counts[i] as f64).powf(inv_temp);
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            0
        } else {
            let r = fast_random_f64() * total;
            let mut cumulative = 0.0;
            let mut chosen = 0u8;
            for i in 0..NUM_ACTIONS {
                cumulative += weights[i];
                if r < cumulative {
                    chosen = i as u8;
                    break;
                }
            }
            chosen
        }
    };

    let root_value = if nodes[0].visit_count > 0 {
        nodes[0].total_value as f32 / nodes[0].visit_count as f32
    } else {
        0.0
    };

    (best_action, policy_target, root_value, root_obs)
}
