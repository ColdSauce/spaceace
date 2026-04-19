use std::cell::Cell;
use std::collections::HashMap;

use crate::pathfinder::PathfinderKind;
use crate::real_game::{GameSnapshot, RealSpaceAceGame};

// Simple fast RNG (xorshift32). Seeded once per thread from the system clock
// so every fresh process gets different tie-breaking — otherwise identical
// MCTS runs collide on the same path every time, which hides variance in
// tight cornering scenarios.
thread_local! {
    static RNG_STATE: Cell<u32> = Cell::new(initial_rng_seed());
}

fn initial_rng_seed() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    // Fold 64-bit nanos into 32 bits and avoid the degenerate 0 state.
    let folded = ((nanos ^ (nanos >> 32)) as u32) | 1;
    folded
}

pub fn set_rng_seed(seed: u32) {
    let seed = if seed == 0 { 1 } else { seed };
    RNG_STATE.with(|state| state.set(seed));
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

#[derive(Clone)]
pub struct MCTSParams {
    pub num_simulations: u32,
    /// Base action_repeat — used at the tree root. Deeper nodes use
    /// action_repeat + depth * action_repeat_depth_bonus (capped at
    /// action_repeat_max). This gives fine control near the root for
    /// precision manoeuvres while still reaching long-horizon lookahead
    /// deep in the tree without growing the tree's branching factor.
    pub action_repeat: u32,
    pub action_repeat_depth_bonus: u32,
    pub action_repeat_max: u32,
    pub c_explore: f64,
    pub gamma: f64,
    pub max_steps: u32,
    pub shaping_weight: f64,
    /// Goofy mode: restrict tree expansion to thrust-on actions (indices 1, 3, 5).
    pub goofy: bool,
    /// Added to the UCT exploitation term of thrust-on children (indices 1, 3, 5).
    /// 0.0 = neutral (default). Small positive values (e.g. 1.0–5.0) bias the tree
    /// toward keeping thrust on without fully forbidding coast/rotate-only moves.
    pub thrust_bias: f64,
    /// Distance (px) at which `thrust_bias` reaches full strength. Below this,
    /// the bias is linearly scaled down by min wall distance so the tree stops
    /// being pushed to thrust when a collision is imminent. 0 disables scaling
    /// (constant bias everywhere). Typical values: 80-200.
    pub thrust_bias_safe_dist: f64,
    /// If true, use PUCT (Q + c·P·√N/(1+n)) with heuristic action priors.
    /// If false (default), use standard UCT — the rule-based priors turned out
    /// worse than the latent priors that UCT gets for free via one-step
    /// heuristic evaluation at each action's first expansion.
    pub use_puct: bool,
    /// Progressive widening coefficient. 0.0 = disabled (all actions always
    /// exposed, original UCT behaviour). When > 0, a node with N visits only
    /// exposes the top ⌈widen_k·√N⌉ actions (ordered by heuristic prior) to
    /// UCT selection. Shrinks the effective branching factor at shallow-visit
    /// nodes so promising lines grow deeper for the same sim budget.
    /// Typical values: 1.0-1.5.
    pub widen_k: f64,
    /// Adaptive early-exit: after every `early_exit_check_every` sims, check
    /// the root visit distribution and stop early if one action clearly
    /// dominates (visit fraction ≥ `early_exit_visit_frac` AND mean-value
    /// gap to runner-up ≥ `early_exit_q_gap`). 0 disables the check.
    ///
    /// Purpose: on easy cruising decisions a handful of sims already settle
    /// the action; the remaining budget is wasted. Early-exit redistributes
    /// that wall-time to hard corner decisions (via the dynamic sim scaling
    /// the Python agent does). Safe defaults are conservative enough that
    /// ambiguous positions still burn the full budget.
    pub early_exit_check_every: u32,
    pub early_exit_visit_frac: f64,
    pub early_exit_q_gap: f64,
}

/// Action indices available when `goofy` is true — thrust is always on.
const GOOFY_ACTIONS: [u8; 3] = [1, 3, 5];

#[inline]
fn is_thrust_action(action_idx: i8) -> bool {
    action_idx == 1 || action_idx == 3 || action_idx == 5
}

/// Linearly scale thrust_bias by how safe the current state is. When the
/// nearest wall is further than `safe_dist`, full bias applies. Closer in,
/// the bias fades linearly to 0 — so MCTS doesn't get pushed to commit thrust
/// into a wall it's seconds from hitting. safe_dist <= 0 disables scaling
/// (constant bias everywhere, original behaviour).
#[inline]
fn scale_thrust_bias(thrust_bias: f64, safe_dist: f64, min_wall_dist: f32) -> f64 {
    if thrust_bias == 0.0 { return 0.0; }
    if safe_dist <= 0.0 { return thrust_bias; }
    let d = min_wall_dist as f64;
    if !d.is_finite() { return thrust_bias; }
    let s = (d / safe_dist).clamp(0.0, 1.0);
    thrust_bias * s
}

/// Minimum of the 8-direction raycast wall-distance array.
#[inline]
fn min_wall_distance(wd: &[f32; 8]) -> f32 {
    wd.iter().copied().fold(f32::INFINITY, f32::min)
}

#[derive(Clone)]
pub struct MCTSNode {
    pub snapshot: GameSnapshot,
    pub step_count: u32,
    /// Tree depth from root — used to scale the effective action_repeat when
    /// expanding children, so deeper tree levels look further into the future
    /// per edge without costing extra branching factor.
    pub depth: u32,
    pub action: i8,        // index into ACTIONS, -1 for root
    pub parent: i32,       // index into nodes vec, -1 for root
    /// child index per action, None if that action has not been expanded yet.
    /// Encoding unvisited actions as first-class entries makes PUCT's
    /// prior-weighted selection uniform across visited and unvisited children.
    pub child_of_action: [i32; NUM_ACTIONS],
    pub visit_count: u32,
    pub total_reward: f64,
    pub is_terminal: bool,
    /// Prior probability P(a) per action, filled in when the node is first
    /// expanded. Sums to 1 over the allowed action set (6 normally, 3 goofy).
    pub priors: [f64; NUM_ACTIONS],
    /// True if priors have been computed on this node yet.
    pub priors_ready: bool,
    /// Minimum wall distance at this state (raycast over 8 directions). Cached
    /// at node creation so thrust_bias scaling is O(1) per selection.
    /// f32::INFINITY means "not yet computed".
    pub min_wall_dist: f32,
    /// Quantized physics+pickups hash used for transposition lookup. Two nodes
    /// with the same key represent physically-equivalent states (same bucket)
    /// and can share value estimates across paths.
    pub state_key: u64,
}

impl MCTSNode {
    fn new_with_mode(snapshot: GameSnapshot, step_count: u32, depth: u32, action: i8, parent: i32, _goofy: bool) -> Self {
        MCTSNode {
            snapshot, step_count, depth, action, parent,
            child_of_action: [-1; NUM_ACTIONS],
            visit_count: 0,
            total_reward: 0.0,
            is_terminal: false,
            priors: [1.0 / NUM_ACTIONS as f64; NUM_ACTIONS],
            priors_ready: false,
            min_wall_dist: f32::INFINITY,
            state_key: 0,
        }
    }

    /// Iterator over currently expanded child node indices.
    pub fn children_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.child_of_action.iter().filter_map(|&c| if c >= 0 { Some(c as usize) } else { None })
    }

    pub fn has_children(&self) -> bool {
        self.child_of_action.iter().any(|&c| c >= 0)
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
    pub route_score: f64,
    pub route_length: f64,
    pub path_dist: f64,
    pub dir_x: f64,
    pub dir_y: f64,
    pub speed_toward: f64,
    pub alignment: f64,
    pub min_tti: f64,
}

/// Weight on remaining-TSP-tour length in the heuristic (world px → score).
/// Calibrated so that ~1000px of remaining route ≈ 100 score units, meaningfully
/// below the +200 bonus for collecting a single pickup — the agent should never
/// be tempted to *avoid* a collection just because the next target is far away,
/// but should prefer routes whose full remaining tour is shorter.
const ROUTE_WEIGHT: f64 = 0.1;

fn evaluate_state_breakdown(game: &RealSpaceAceGame, pathfinder: &PathfinderKind) -> HeuristicBreakdown {
    let mut bd = HeuristicBreakdown {
        total: 0.0, pickups_score: 0.0, proximity_score: 0.0,
        velocity_score: 0.0, orientation_score: 0.0, tangential_penalty: 0.0,
        tti_penalty: 0.0, route_score: 0.0, route_length: 0.0,
        path_dist: 0.0, dir_x: 0.0, dir_y: 0.0,
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

    // Full remaining-tour length (greedy TSP). Penalize longer tours so MCTS
    // prefers positions/routes that reduce the *entire* remaining trip, not
    // just distance to the nearest pickup. Only adds a negative contribution
    // (clamped at 0 when no pickups remain), so collecting a pickup always
    // strictly improves the total.
    bd.route_length = pathfinder.get_remaining_route_length(ship_x, ship_y, ship_vx, ship_vy, &collected);
    bd.route_score = -ROUTE_WEIGHT * bd.route_length;

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

    // Total speed magnitude (used by velocity bonus and TTI scaling below).
    let speed = ((ship_vx * ship_vx + ship_vy * ship_vy) as f64).sqrt();

    // Route tangent: direction the ship should be moving to stay on the
    // optimal path, looking ~8 cells ahead along the BFS gradient. Differs
    // from `guide_dx/dy` (direction to current target pickup) when the route
    // bends around a wall — the tangent follows the corridor, the pickup
    // vector would point through the wall. Using the tangent for velocity
    // and orientation lets the heuristic reward expert corner play (where
    // the ship briefly points away from the pickup to carry momentum
    // through the turn).
    //
    // Near the pickup (`guide_dist < ~100px`), the tangent collapses onto
    // the pickup direction anyway (path and euclidean converge), so there's
    // no behavior regression in the approach phase.
    let (route_tan_x, route_tan_y, route_tan_ok) =
        pathfinder.get_route_tangent(ship_x, ship_y, ship_vx, ship_vy, &collected, 8);
    // Pick the reference direction once for velocity + orientation.
    let (ref_dx, ref_dy) = if route_tan_ok {
        (route_tan_x, route_tan_y)
    } else {
        (guide_dx, guide_dy)
    };

    // Velocity reward — aggressive by design. The raw-speed bonus is on
    // everywhere (no distance gating) and the caps are raised well above
    // typical gameplay, so carrying momentum through pickups is always
    // rewarded. An overshoot penalty survives but at a fraction of the
    // original weight: just enough to prefer a brakeable approach over one
    // that can't possibly stop, not enough to discourage swooping.
    if guide_dist > 0.0 {
        bd.speed_toward = ship_vx as f64 * ref_dx + ship_vy as f64 * ref_dy;
        let toward = bd.speed_toward.max(-200.0).min(500.0) * 1.4;

        let mut overshoot_penalty = 0.0;
        if bd.speed_toward > 80.0 {
            let target_speed = (2.0 * 205.0 * guide_dist).sqrt();
            let excess = (bd.speed_toward - target_speed).max(0.0);
            overshoot_penalty = -(excess.min(250.0)) * 0.5;
        }

        let total_speed_bonus = speed.min(500.0) * 0.6;

        bd.velocity_score = toward + overshoot_penalty + total_speed_bonus;
    }

    // Ship orientation alignment — only matters far from pickup (approach heading).
    // Near the pickup we care about position/velocity, not orientation, since
    // turning through a corner at speed requires briefly pointing away.
    if guide_dist > 0.0 {
        let heading_x = ship_rot.sin();
        let heading_y = -ship_rot.cos();
        bd.alignment = heading_x * ref_dx + heading_y * ref_dy;
        let orient_weight = (guide_dist / 400.0).min(1.0) * 8.0;
        bd.orientation_score = bd.alignment * orient_weight;
    }

    // TTI wall avoidance
    let wall_distances = game.get_wall_distances();
    let cos_r = ship_rot.cos();
    let sin_r = ship_rot.sin();
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

    // Scale TTI threshold with speed: faster ships need to react earlier.
    // Now that the velocity rewards are uncapped and the agent is expected
    // to fly at 400+ px/s, the old 0.6s cap was too short — give it more
    // reaction headroom and a steeper penalty so high-speed crash lines
    // still lose out to equally-fast non-crash lines.
    let tti_threshold = 0.27 + (speed / 280.0).min(0.50);
    if bd.min_tti < tti_threshold {
        let deficit = tti_threshold - bd.min_tti;
        bd.tti_penalty = -deficit * deficit * 725.0;
        if bd.min_tti < 0.2 {
            bd.tti_penalty -= 725.0;
        }
    }

    // Direct wall proximity penalty: only triggered when very close (< 25px),
    // and only scaled by speed toward walls (not total speed). Corners require
    // passing close to walls at speed; penalizing that directly = braking.
    if min_wall_dist < 25.0 {
        let prox_penalty = (25.0 - min_wall_dist) / 25.0; // 0..1
        bd.tti_penalty -= prox_penalty * prox_penalty * 100.0;
    }

    bd.total = bd.pickups_score + bd.proximity_score + bd.velocity_score
        + bd.orientation_score + bd.tti_penalty + bd.route_score;
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

/// Compute action priors for a node given its game state. Cheap rule-based
/// policy: combines alignment with the pickup direction (for thrust/coast),
/// cross product with the heading (for rotation direction), and an overshoot
/// check (suppresses thrust when already moving too fast to brake). The
/// returned array is a probability distribution; goofy mode zeroes out the
/// three non-thrust actions and renormalises.
fn compute_action_priors(game: &RealSpaceAceGame, pathfinder: &PathfinderKind, goofy: bool) -> [f64; NUM_ACTIONS] {
    let state = game.get_state();
    let collected: Vec<bool> = game.get_pickup_positions().iter().map(|&(_, _, c)| c).collect();
    let (dist, dir_x, dir_y) = pathfinder.get_nearest_pickup_info(
        state.ship_x, state.ship_y, state.ship_vx, state.ship_vy, &collected);

    let rot = state.ship_rotation as f64;
    let heading_x = rot.sin();
    let heading_y = -rot.cos();
    let alignment = heading_x * dir_x + heading_y * dir_y;
    let cross = heading_x * dir_y - heading_y * dir_x; // >0 → target on right

    let speed_toward = state.ship_vx as f64 * dir_x + state.ship_vy as f64 * dir_y;
    let can_overshoot = dist > 0.0 && speed_toward > (2.0 * 205.0 * dist.max(1.0)).sqrt();

    let thrust_desire: f64 = if can_overshoot {
        -1.5
    } else {
        // +2 when perfectly aligned and not overshooting, scaled by how
        // much headroom we have before the brake threshold.
        alignment * 2.0
    };
    let rot_left_desire = ((-cross).max(0.0) * 2.0 - 0.3).max(-1.0);
    let rot_right_desire = (cross.max(0.0) * 2.0 - 0.3).max(-1.0);
    let coast_desire = if alignment > 0.7 && speed_toward > 100.0 { 0.5 } else { -0.5 };

    let mut logits = [0.0f64; NUM_ACTIONS];
    logits[0] = coast_desire;                                 // coast
    logits[1] = thrust_desire;                                // thrust
    logits[2] = rot_left_desire;                              // rotate-left
    logits[3] = rot_left_desire + thrust_desire * 0.5;        // rotate-left + thrust
    logits[4] = rot_right_desire;                             // rotate-right
    logits[5] = rot_right_desire + thrust_desire * 0.5;       // rotate-right + thrust

    if goofy {
        // Mask non-thrust actions (indices 0, 2, 4).
        logits[0] = f64::NEG_INFINITY;
        logits[2] = f64::NEG_INFINITY;
        logits[4] = f64::NEG_INFINITY;
    }

    // Softmax with numerical stability.
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps = [0.0f64; NUM_ACTIONS];
    let mut exp_sum = 0.0;
    for i in 0..NUM_ACTIONS {
        exps[i] = if logits[i].is_finite() { (logits[i] - max_logit).exp() } else { 0.0 };
        exp_sum += exps[i];
    }
    let mut priors = [0.0f64; NUM_ACTIONS];
    if exp_sum > 0.0 {
        for i in 0..NUM_ACTIONS {
            priors[i] = exps[i] / exp_sum;
        }
    } else {
        // fallback: uniform over the allowed action set
        let allowed: &[u8] = if goofy { &GOOFY_ACTIONS } else { &[0, 1, 2, 3, 4, 5] };
        for &a in allowed {
            priors[a as usize] = 1.0 / allowed.len() as f64;
        }
    }
    priors
}

/// PUCT selection: picks the action that maximises
///   Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + n(a))
/// treating unexpanded actions as n(a)=0 with Q=FPU (first-play urgency=0).
fn best_puct_action(
    parent: &MCTSNode,
    nodes: &[MCTSNode],
    c_puct: f64,
    thrust_bias: f64,
    thrust_bias_safe_dist: f64,
    goofy: bool,
) -> u8 {
    let parent_visits_sqrt = (parent.visit_count as f64).sqrt().max(1.0);
    let allowed: &[u8] = if goofy { &GOOFY_ACTIONS } else { &[0, 1, 2, 3, 4, 5] };
    let effective_thrust_bias = scale_thrust_bias(thrust_bias, thrust_bias_safe_dist, parent.min_wall_dist);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_action = allowed[0];
    for &a in allowed {
        let child_idx = parent.child_of_action[a as usize];
        let (q, n_a) = if child_idx >= 0 {
            let c = &nodes[child_idx as usize];
            let q = if c.visit_count > 0 { c.total_reward / c.visit_count as f64 } else { 0.0 };
            (q, c.visit_count as f64)
        } else {
            (0.0, 0.0)
        };
        let u = c_puct * parent.priors[a as usize] * parent_visits_sqrt / (1.0 + n_a);
        let bias = if effective_thrust_bias != 0.0 && is_thrust_action(a as i8) { effective_thrust_bias } else { 0.0 };
        let score = q + u + bias;
        if score > best_score {
            best_score = score;
            best_action = a;
        }
    }
    best_action
}

/// Standard UCT selection on the new child_of_action layout, with optional
/// progressive widening. When `widen_k > 0`, only the top ⌈widen_k·√N⌉
/// actions (ordered by prior, descending) are "exposed" and considered for
/// expansion / UCT; the rest are invisible until N grows. When `widen_k == 0`
/// all actions are always exposed, giving the original full-branching UCT.
fn best_uct_action(
    parent: &MCTSNode,
    nodes: &[MCTSNode],
    c_explore: f64,
    thrust_bias: f64,
    thrust_bias_safe_dist: f64,
    goofy: bool,
    widen_k: f64,
) -> u8 {
    let effective_thrust_bias = scale_thrust_bias(thrust_bias, thrust_bias_safe_dist, parent.min_wall_dist);
    let allowed_arr: [u8; 6] = [0, 1, 2, 3, 4, 5];
    let allowed: &[u8] = if goofy { &GOOFY_ACTIONS } else { &allowed_arr };
    let n_allowed = allowed.len();

    // Order allowed actions by prior (descending). With widen_k == 0 the
    // ordering is irrelevant for selection, but we still use a random
    // tie-breaking order so a node with no priors computed doesn't collapse
    // into fully deterministic play.
    let mut ordered: [u8; 6] = [0; 6];
    for (i, &a) in allowed.iter().enumerate() { ordered[i] = a; }
    if widen_k > 0.0 && parent.priors_ready {
        // Insertion sort by prior descending (n<=6, tiny).
        for i in 1..n_allowed {
            let key = ordered[i];
            let key_prior = parent.priors[key as usize];
            let mut j = i;
            while j > 0 && parent.priors[ordered[j - 1] as usize] < key_prior {
                ordered[j] = ordered[j - 1];
                j -= 1;
            }
            ordered[j] = key;
        }
    } else {
        // Fisher-Yates shuffle for random tie-breaking.
        for i in (1..n_allowed).rev() {
            let j = fast_random() as usize % (i + 1);
            ordered.swap(i, j);
        }
    }

    // How many actions are exposed right now?
    let exposed = if widen_k > 0.0 {
        let k = (widen_k * ((parent.visit_count as f64) + 1.0).sqrt()).ceil() as usize;
        k.max(1).min(n_allowed)
    } else {
        n_allowed
    };
    let exposed_actions = &ordered[..exposed];

    // Try any unexpanded exposed action first (UCT's +∞ on unvisited children).
    for &a in exposed_actions {
        if parent.child_of_action[a as usize] < 0 {
            return a;
        }
    }

    // All exposed actions are expanded — UCT among them.
    let parent_log = (parent.visit_count as f64).ln().max(0.0);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_action = exposed_actions[0];
    for &a in exposed_actions {
        let child_idx = parent.child_of_action[a as usize] as usize;
        let c = &nodes[child_idx];
        let q = c.total_reward / c.visit_count as f64;
        let u = c_explore * (parent_log / c.visit_count as f64).sqrt();
        let bias = if effective_thrust_bias != 0.0 && is_thrust_action(a as i8) { effective_thrust_bias } else { 0.0 };
        let score = q + u + bias;
        if score > best_score {
            best_score = score;
            best_action = a;
        }
    }
    best_action
}

// ---------------------------------------------------------------------------
// Transposition-table state keying.
//
// Quantizes the continuous physics state plus the discrete pickups bitmask
// into a single u64. States within the same bucket are treated as equivalent
// for value-sharing purposes: when a simulation would create a new child whose
// key is already in the tree's transposition map, the existing node is reused
// (DAG-style edge) and its cached mean value is used as this leaf's evaluation.
//
// Quantization buckets are deliberately coarser than fine physics precision.
// Too-fine => ~zero transposition hits (states never repeat exactly). Too-
// coarse => incorrect value sharing (states in the same bucket behave
// differently). The numbers below are tuned for this game's typical speeds
// (200-500 px/s) and spatial scale (10-20 px grid cells in the pathfinder).
// ---------------------------------------------------------------------------

const STATE_KEY_XY_BUCKET: f32 = 6.0;      // px  — below ship radius, roughly one pathfinder cell
const STATE_KEY_V_BUCKET: f32 = 25.0;      // px/s
const STATE_KEY_ROT_BUCKETS: f64 = 64.0;   // 64 angular buckets across 2π (~5.6°)

fn compute_state_key(snap: &GameSnapshot) -> u64 {
    let qx = (snap.physics.x / STATE_KEY_XY_BUCKET).round() as i32 as i64;
    let qy = (snap.physics.y / STATE_KEY_XY_BUCKET).round() as i32 as i64;
    let qvx = (snap.physics.vx / STATE_KEY_V_BUCKET).round() as i32 as i64;
    let qvy = (snap.physics.vy / STATE_KEY_V_BUCKET).round() as i32 as i64;
    let two_pi = std::f64::consts::TAU;
    let rot = (snap.physics.rotation as f64).rem_euclid(two_pi);
    let qrot = ((rot / two_pi) * STATE_KEY_ROT_BUCKETS).floor() as i64;
    let mut pickup_mask: u64 = 0;
    for (i, p) in snap.pickups.iter().enumerate().take(64) {
        if p.collected { pickup_mask |= 1u64 << i; }
    }

    // Mix fields with a SplitMix64-style step so collisions are rare for
    // nearby buckets. (HashMap<u64, _> will re-hash internally.)
    let mut h: u64 = 0xcbf29ce484222325;
    let fold = |h: &mut u64, v: i64| {
        *h ^= v as u64;
        *h = h.wrapping_mul(0x100000001b3);
    };
    fold(&mut h, qx); fold(&mut h, qy);
    fold(&mut h, qvx); fold(&mut h, qvy);
    fold(&mut h, qrot);
    h ^= pickup_mask;
    h = h.wrapping_mul(0x100000001b3);
    if snap.physics.exploded { h ^= 0xdead_beef_dead_beef; }
    h
}

/// Tree container for MCTS — supports incremental growth and re-rooting between decisions.
#[derive(Clone)]
pub struct MCTSTree {
    pub nodes: Vec<MCTSNode>,
    /// Heuristic at tree root; values are stored as (reward + gamma*(leaf - root_baseline))
    /// so that UCT exploitation is centered near zero for normal states. Must be
    /// re-normalized on re-root so old and new sims share a baseline.
    pub root_baseline: f64,
    pub goofy: bool,
    /// Transposition map: quantized state_key → node index. Reused across
    /// simulations to link new "actions" back to existing physically-equivalent
    /// nodes. Rebuilt on re-root.
    pub transpositions: HashMap<u64, usize>,
    /// Counters for instrumenting transposition-hit rate.
    pub transposition_hits: u64,
    pub transposition_misses: u64,
}

impl MCTSTree {
    pub fn new(
        snapshot: GameSnapshot,
        step_count: u32,
        game: &mut RealSpaceAceGame,
        pathfinder: &PathfinderKind,
        goofy: bool,
    ) -> Self {
        game.load_state(snapshot.clone());
        let root_baseline = evaluate_state(game, pathfinder);
        let root_wall_dist = min_wall_distance(&game.get_wall_distances());
        let root_key = compute_state_key(&snapshot);
        let mut root = MCTSNode::new_with_mode(snapshot, step_count, 0, -1, -1, goofy);
        root.min_wall_dist = root_wall_dist;
        root.state_key = root_key;
        let mut transpositions = HashMap::new();
        transpositions.insert(root_key, 0);
        MCTSTree {
            nodes: vec![root],
            root_baseline,
            goofy,
            transpositions,
            transposition_hits: 0,
            transposition_misses: 0,
        }
    }

    pub fn root_snapshot(&self) -> &GameSnapshot {
        &self.nodes[0].snapshot
    }

    /// Run a batch of MCTS simulations, growing the existing tree.
    ///
    /// Respects `params.early_exit_check_every`: when > 0, checks the root
    /// action distribution after each batch and exits early if the best
    /// action dominates by both visits and mean value. When 0 (the default
    /// for legacy callers), runs the full budget.
    pub fn run_simulations(
        &mut self,
        game: &mut RealSpaceAceGame,
        pathfinder: &PathfinderKind,
        params: &MCTSParams,
    ) -> u32 {
        let total = params.num_simulations;
        let check_every = params.early_exit_check_every;
        if check_every == 0 {
            for _ in 0..total {
                self.run_one_sim(game, pathfinder, params);
            }
            return total;
        }

        // Require at least 2 full check windows before the first exit eval so
        // we don't early-exit on a few biased sims.
        let min_before_exit = check_every.saturating_mul(2);

        let mut done: u32 = 0;
        while done < total {
            let batch = check_every.min(total - done);
            for _ in 0..batch {
                self.run_one_sim(game, pathfinder, params);
            }
            done += batch;
            if done >= min_before_exit && self.should_early_exit(params) {
                break;
            }
        }
        done
    }

    /// Dominance test used by adaptive early-exit. Returns true iff the
    /// highest-visit root child meets BOTH thresholds:
    ///   visit_fraction ≥ params.early_exit_visit_frac
    ///   q_gap_to_second_best ≥ params.early_exit_q_gap
    /// Checking both avoids two pitfalls:
    ///   - High visits with a tiny Q-gap → UCT still actively exploring.
    ///   - Large Q-gap with few visits → outlier sim, not yet confirmed.
    fn should_early_exit(&self, params: &MCTSParams) -> bool {
        if self.nodes.is_empty() { return false; }
        let root = &self.nodes[0];
        let mut total: u32 = 0;
        let mut best_a: usize = 0;
        let mut best_v: u32 = 0;
        let mut best_q: f64 = f64::NEG_INFINITY;
        let mut second_q: f64 = f64::NEG_INFINITY;
        // First pass: find best by visits
        for a in 0..NUM_ACTIONS {
            let idx = root.child_of_action[a];
            if idx < 0 { continue; }
            let c = &self.nodes[idx as usize];
            total += c.visit_count;
            if c.visit_count > best_v {
                best_v = c.visit_count;
                best_a = a;
            }
        }
        if total == 0 || best_v == 0 { return false; }
        let best_frac = best_v as f64 / total as f64;
        if best_frac < params.early_exit_visit_frac { return false; }

        // Second pass: Q of best, and best-Q among the rest.
        for a in 0..NUM_ACTIONS {
            let idx = root.child_of_action[a];
            if idx < 0 { continue; }
            let c = &self.nodes[idx as usize];
            if c.visit_count == 0 { continue; }
            let q = c.total_reward / c.visit_count as f64;
            if a == best_a { best_q = q; }
            else if q > second_q { second_q = q; }
        }
        if !best_q.is_finite() || !second_q.is_finite() { return false; }
        (best_q - second_q) >= params.early_exit_q_gap
    }

    fn run_one_sim(
        &mut self,
        game: &mut RealSpaceAceGame,
        pathfinder: &PathfinderKind,
        params: &MCTSParams,
    ) {
        let mut node_idx: usize = 0;
        let mut reward: f64 = 0.0;
        // Actual nodes visited this simulation (in descent order). Backprop walks
        // this in reverse instead of chasing `parent` pointers, so transposition
        // edges (which link to a node whose `parent` was set by its *original*
        // discoverer, not by this path) still get correct visit/value updates
        // along the path we actually took.
        let mut path: Vec<usize> = vec![0];
        // Set of nodes in `path` for O(1) cycle-check. Small (≤ ~depth), so a
        // linear scan would be fine, but keeping a set makes the intent explicit.
        let mut in_path: std::collections::HashSet<usize> = std::collections::HashSet::new();
        in_path.insert(0);
        // If we descend into a transposed node, we skip leaf re-evaluation and
        // instead use its cached mean value (the point of a transposition table).
        // `leaf_value_override` carries that value past the selection loop.
        let mut leaf_value_override: Option<f64> = None;

        // --- SELECTION + EXPANSION ---
        loop {
            if self.nodes[node_idx].is_terminal { break; }

            // Compute priors lazily on first visit when either PUCT or
            // progressive widening needs them to order actions.
            if (params.use_puct || params.widen_k > 0.0) && !self.nodes[node_idx].priors_ready {
                game.load_state(self.nodes[node_idx].snapshot.clone());
                self.nodes[node_idx].priors = compute_action_priors(game, pathfinder, self.goofy);
                self.nodes[node_idx].priors_ready = true;
            }

            let action = if params.use_puct {
                best_puct_action(
                    &self.nodes[node_idx],
                    &self.nodes,
                    params.c_explore,
                    params.thrust_bias,
                    params.thrust_bias_safe_dist,
                    self.goofy,
                )
            } else {
                best_uct_action(
                    &self.nodes[node_idx],
                    &self.nodes,
                    params.c_explore,
                    params.thrust_bias,
                    params.thrust_bias_safe_dist,
                    self.goofy,
                    params.widen_k,
                )
            };

            let existing_child = self.nodes[node_idx].child_of_action[action as usize];
            if existing_child >= 0 {
                let c = existing_child as usize;
                // Cycle guard: if we've already visited this node in this sim,
                // stop descending and evaluate the current node as a leaf.
                // Transposition links can create cycles in principle (path
                // loops back to a shallower state); treating the revisit as a
                // terminal leaf caps sim cost at O(tree depth).
                if in_path.contains(&c) { break; }
                path.push(c);
                in_path.insert(c);
                node_idx = c;
                continue;
            }

            // Expand: simulate `action_repeat` frames of the chosen action.
            // Scale action_repeat with tree depth: fine control near the root
            // (precise cornering), larger macro-actions deeper in the tree
            // (long-horizon strategic lookahead).
            let parent_depth = self.nodes[node_idx].depth;
            let effective_ar = (params.action_repeat
                + parent_depth.saturating_mul(params.action_repeat_depth_bonus))
                .min(params.action_repeat_max.max(params.action_repeat));
            let action_controls = ACTIONS[action as usize];
            game.load_state(self.nodes[node_idx].snapshot.clone());

            // Reward shaping: path distance before/after the macro-action.
            let collected_before: Vec<bool> = game.get_pickup_positions().iter().map(|&(_, _, c)| c).collect();
            let state_before = game.get_state();
            let (dist_before, _, _) = pathfinder.get_nearest_pickup_info(
                state_before.ship_x, state_before.ship_y,
                state_before.ship_vx, state_before.ship_vy, &collected_before);

            let mut step_count = self.nodes[node_idx].step_count;
            let mut terminated = false;
            let mut truncated = false;
            for _ in 0..effective_ar {
                game.set_controls(action_controls[0], action_controls[1], action_controls[2]);
                game.step(1.0 / 60.0);
                step_count += 1;
                reward += crate::calculate_reward(game) as f64;
                terminated = game.is_terminated();
                truncated = step_count >= params.max_steps;
                if terminated || truncated { break; }
            }
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
            let is_terminal_child = terminated || truncated;
            let child_key = compute_state_key(&child_snapshot);

            // Transposition lookup: if another path already reached this
            // (bucketed) state, reuse the existing node instead of creating a
            // fresh one. Skip if it would form a cycle with this sim's path,
            // or if the existing node is non-terminal but this edge reached a
            // terminal state (terminal status is path-specific via step_count
            // truncation, so don't merge across that boundary).
            if !is_terminal_child {
                if let Some(&existing_idx) = self.transpositions.get(&child_key) {
                    if existing_idx != node_idx
                        && !in_path.contains(&existing_idx)
                        && !self.nodes[existing_idx].is_terminal
                    {
                        // Link parent → existing, and stop: use the existing
                        // node's cached mean value as this leaf's eval.
                        self.nodes[node_idx].child_of_action[action as usize] = existing_idx as i32;
                        path.push(existing_idx);
                        // Not inserted into in_path (we break immediately).
                        let en = &self.nodes[existing_idx];
                        let mean = if en.visit_count > 0 {
                            en.total_reward / en.visit_count as f64
                        } else {
                            0.0
                        };
                        leaf_value_override = Some(mean);
                        self.transposition_hits += 1;
                        break;
                    }
                }
            }

            let child_wall_dist = min_wall_distance(&game.get_wall_distances());
            let new_idx = self.nodes.len();
            let mut child = MCTSNode::new_with_mode(child_snapshot, step_count, parent_depth + 1, action as i8, node_idx as i32, self.goofy);
            child.is_terminal = is_terminal_child;
            child.min_wall_dist = child_wall_dist;
            child.state_key = child_key;
            self.nodes.push(child);
            self.nodes[node_idx].child_of_action[action as usize] = new_idx as i32;
            // Only record the first discoverer as the canonical entry for this
            // key — subsequent expansions that produce the same key will link
            // to this node via the branch above. Terminal nodes are intention-
            // ally not recorded (see above).
            if !is_terminal_child {
                self.transpositions.entry(child_key).or_insert(new_idx);
            }
            self.transposition_misses += 1;
            path.push(new_idx);
            node_idx = new_idx;
            break;
        }

        // --- EVALUATION ---
        let value = match leaf_value_override {
            Some(v) => reward + params.gamma * v,
            None => {
                game.load_state(self.nodes[node_idx].snapshot.clone());
                let heuristic = evaluate_state(game, pathfinder);
                reward + params.gamma * (heuristic - self.root_baseline)
            }
        };

        // --- BACKPROPAGATION along the actually-traversed path ---
        for &i in path.iter().rev() {
            self.nodes[i].visit_count += 1;
            self.nodes[i].total_reward += value;
        }
    }

    /// Return (best_action, [(action, visits, mean_value)], root_baseline).
    pub fn best_action_and_stats(&self) -> (u8, Vec<(u8, u32, f64)>, f64) {
        let nodes = &self.nodes;
        // Iterate over the root's own `child_of_action` slots rather than via
        // `children_iter`+`child.action`. With the transposition DAG, the root
        // may link to a node whose `.action` field was recorded when a *different*
        // parent first created it — using that field would mislabel the stats
        // (and produce out-of-range action ids when the target was the root itself
        // and its action is -1). The root's slot index IS the action taken from
        // the root, by construction, so it's the only correct key here.
        let mut action_stats: Vec<(u8, u32, f64)> = Vec::new();
        for a in 0..NUM_ACTIONS {
            let idx = nodes[0].child_of_action[a];
            if idx < 0 { continue; }
            let child = &nodes[idx as usize];
            let mean_val = if child.visit_count > 0 {
                child.total_reward / child.visit_count as f64
            } else {
                0.0
            };
            action_stats.push((a as u8, child.visit_count, mean_val));
        }

        let best_action = if !nodes[0].has_children() {
            1
        } else {
            let mut best_action = 1u8;
            let mut best_visits = 0u32;
            for a in 0..NUM_ACTIONS {
                let idx = nodes[0].child_of_action[a];
                if idx >= 0 {
                    let v = nodes[idx as usize].visit_count;
                    if v > best_visits {
                        best_visits = v;
                        best_action = a as u8;
                    }
                }
            }
            best_action
        };

        (best_action, action_stats, self.root_baseline)
    }

    /// Try to re-root the tree at the child of the current root reached by
    /// `action`, but only if that child's snapshot matches `target_snapshot`.
    ///
    /// On success, re-normalises stored values so they are relative to the
    /// new root's heuristic baseline (keeps UCT exploitation centred).
    ///
    /// Returns true if reuse succeeded, false if the caller should rebuild.
    pub fn try_reroot_to_action(
        &mut self,
        action: u8,
        target_snapshot: &GameSnapshot,
        game: &mut RealSpaceAceGame,
        pathfinder: &PathfinderKind,
        gamma: f64,
    ) -> bool {
        if self.nodes.is_empty() { return false; }
        let root = 0;
        let child_raw = self.nodes[root].child_of_action[action as usize];
        if child_raw < 0 { return false; }
        let child_old_idx = child_raw as usize;

        if !snapshots_equal(&self.nodes[child_old_idx].snapshot, target_snapshot) {
            return false;
        }

        // Extract subtree rooted at child_old_idx into a fresh vec with re-indexed pointers.
        let new_nodes = extract_subtree(&self.nodes, child_old_idx);

        // Compute new baseline from the new root's state.
        game.load_state(target_snapshot.clone());
        let new_baseline = evaluate_state(game, pathfinder);

        // Re-normalise stored total_reward so new and old sims share a baseline.
        // value = reward + gamma * (leaf - baseline), so shifting baseline by Δ
        // shifts each stored value by -gamma·Δ per visit.
        let delta = gamma * (self.root_baseline - new_baseline);
        self.nodes = new_nodes;
        if delta.abs() > 1e-9 {
            for node in &mut self.nodes {
                node.total_reward += node.visit_count as f64 * delta;
            }
        }
        // Rebuild the transposition map for the surviving subtree. Indices
        // have been re-numbered by extract_subtree, so the old map is useless.
        self.transpositions.clear();
        for (i, node) in self.nodes.iter().enumerate() {
            if !node.is_terminal {
                self.transpositions.entry(node.state_key).or_insert(i);
            }
        }
        self.root_baseline = new_baseline;
        true
    }

    pub fn total_nodes(&self) -> usize { self.nodes.len() }
    pub fn root_visits(&self) -> u32 {
        if self.nodes.is_empty() { 0 } else { self.nodes[0].visit_count }
    }
}

fn snapshots_equal(a: &GameSnapshot, b: &GameSnapshot) -> bool {
    if a.pickups.len() != b.pickups.len() { return false; }
    // Physics state — deterministic stepping, so exact float equality is OK.
    if a.physics.x != b.physics.x || a.physics.y != b.physics.y { return false; }
    if a.physics.vx != b.physics.vx || a.physics.vy != b.physics.vy { return false; }
    if a.physics.rotation != b.physics.rotation { return false; }
    if a.physics.exploded != b.physics.exploded { return false; }
    if a.pickups_collected_this_step != b.pickups_collected_this_step { return false; }
    for (p, q) in a.pickups.iter().zip(b.pickups.iter()) {
        if p.collected != q.collected { return false; }
    }
    true
}

fn extract_subtree(old_nodes: &[MCTSNode], new_root_old_idx: usize) -> Vec<MCTSNode> {
    // DFS from new_root_old_idx collecting reachable nodes. The tree is now
    // a DAG (transposition edges), so a plain stack without visited-tracking
    // would revisit shared descendants exponentially. Dedupe with a HashSet.
    let mut order: Vec<usize> = Vec::new();
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut stack = vec![new_root_old_idx];
    while let Some(idx) = stack.pop() {
        if !seen.insert(idx) { continue; }
        order.push(idx);
        for child in old_nodes[idx].children_iter() {
            if !seen.contains(&child) { stack.push(child); }
        }
    }
    let mut old_to_new: HashMap<usize, usize> = HashMap::with_capacity(order.len());
    for (new_idx, &old_idx) in order.iter().enumerate() {
        old_to_new.insert(old_idx, new_idx);
    }
    let mut new_nodes: Vec<MCTSNode> = Vec::with_capacity(order.len());
    for &old_idx in &order {
        let mut node = old_nodes[old_idx].clone();
        if old_idx == new_root_old_idx {
            node.parent = -1;
            node.action = -1;
        } else {
            // Parent may be outside the subtree if this node was reached here
            // solely via a transposition link (the node's original discoverer
            // lived in the pruned region). Fall back to -1 in that case —
            // backprop no longer relies on parent pointers anyway.
            node.parent = old_to_new.get(&(node.parent as usize))
                .map(|&i| i as i32).unwrap_or(-1);
        }
        // Re-index child_of_action using the subtree map; entries outside the
        // subtree (shouldn't happen for a contiguous tree, but be defensive)
        // become -1.
        for a in 0..NUM_ACTIONS {
            let c = node.child_of_action[a];
            node.child_of_action[a] = if c >= 0 {
                old_to_new.get(&(c as usize)).map(|&i| i as i32).unwrap_or(-1)
            } else {
                -1
            };
        }
        new_nodes.push(node);
    }
    new_nodes
}

// ---------------------------------------------------------------------------
// Backwards-compatible entry points (create a fresh tree each call).
// ---------------------------------------------------------------------------

pub fn mcts_search(
    game: &mut RealSpaceAceGame,
    pathfinder: &PathfinderKind,
    root_snapshot: GameSnapshot,
    root_step_count: u32,
    params: &MCTSParams,
) -> u8 {
    let mut tree = MCTSTree::new(root_snapshot, root_step_count, game, pathfinder, params.goofy);
    tree.nodes.reserve(params.num_simulations as usize + 16);
    tree.run_simulations(game, pathfinder, params);
    tree.best_action_and_stats().0
}

/// Root-parallel MCTS (Chaslot 2008).
///
/// Runs `num_threads` independent trees in parallel, each receiving
/// ceil(num_simulations / num_threads) simulations. Root action statistics are
/// then merged by summing visit counts and taking a visit-weighted average of
/// mean values — the standard "majority of voters" aggregation.
///
/// This sacrifices the within-tree information sharing that full tree-parallel
/// MCTS would provide (different threads explore the same state from scratch,
/// N times over), but it is lock-free, scales linearly until wall time is
/// dominated by game-step cost, and needs no shared mutable tree state — just
/// a per-thread clone of the game simulator. For MCTS with strong heuristic
/// leaf eval, the independent-trees variance averages out well.
pub fn mcts_search_parallel(
    game_template: &RealSpaceAceGame,
    pathfinder: &PathfinderKind,
    root_snapshot: GameSnapshot,
    root_step_count: u32,
    params: &MCTSParams,
    num_threads: u32,
) -> (u8, Vec<(u8, u32, f64)>, f64) {
    use rayon::prelude::*;

    let nt = num_threads.max(1);
    // Distribute sims, rounding up so total >= requested.
    let sims_per_thread = (params.num_simulations + nt - 1) / nt;

    let per_thread: Vec<(Vec<(u8, u32, f64)>, f64)> = (0..nt).into_par_iter().map(|_| {
        let mut g = game_template.clone();
        let mut tree = MCTSTree::new(
            root_snapshot.clone(), root_step_count, &mut g, pathfinder, params.goofy);
        tree.nodes.reserve(sims_per_thread as usize);
        let mut local = params.clone();
        local.num_simulations = sims_per_thread;
        tree.run_simulations(&mut g, pathfinder, &local);
        let (_best, stats, baseline) = tree.best_action_and_stats();
        (stats, baseline)
    }).collect();

    // Merge by summing visits; combine means by visit-weighted average. Each
    // tree estimated its own mean against its own baseline — but because all
    // trees share the same root snapshot and pathfinder, their baselines
    // coincide within floating-point noise, so weighted averaging is sound.
    let mut visits = [0u32; NUM_ACTIONS];
    let mut weighted_sum = [0.0f64; NUM_ACTIONS];
    let mut baseline_accum = 0.0;
    for (stats, baseline) in &per_thread {
        baseline_accum += baseline;
        for &(a, v, mean) in stats {
            let i = a as usize;
            visits[i] += v;
            weighted_sum[i] += mean * v as f64;
        }
    }
    let merged_baseline = baseline_accum / nt as f64;
    let mut merged: Vec<(u8, u32, f64)> = Vec::new();
    let mut best_a: u8 = 1;
    let mut best_v: u32 = 0;
    for a in 0..NUM_ACTIONS {
        if visits[a] > 0 {
            let mean = weighted_sum[a] / visits[a] as f64;
            merged.push((a as u8, visits[a], mean));
            if visits[a] > best_v {
                best_v = visits[a];
                best_a = a as u8;
            }
        }
    }
    (best_a, merged, merged_baseline)
}

/// Like mcts_search but also returns per-action stats: (best_action, [(action_idx, visits, mean_value)], root_heuristic)
pub fn mcts_search_with_stats(
    game: &mut RealSpaceAceGame,
    pathfinder: &PathfinderKind,
    root_snapshot: GameSnapshot,
    root_step_count: u32,
    params: &MCTSParams,
) -> (u8, Vec<(u8, u32, f64)>, f64) {
    let mut tree = MCTSTree::new(root_snapshot, root_step_count, game, pathfinder, params.goofy);
    tree.nodes.reserve(params.num_simulations as usize + 16);
    tree.run_simulations(game, pathfinder, params);
    tree.best_action_and_stats()
}
