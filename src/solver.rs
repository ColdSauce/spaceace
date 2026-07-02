//! Ace solver: offline time-optimal planner for SpaceAce levels.
//!
//! This is the entire AI in one file. It replaces the old MCTS / A* / beam /
//! kinodyn / polish-script zoo with three pieces:
//!
//!   1. An exact, allocation-free re-implementation of the game step
//!      (`SimState` + `AceSolver::step`). It mirrors `real_physics.rs` /
//!      `real_game.rs` float-op for float-op, so any action tape found here
//!      replays identically on `PyGameInstance`.
//!   2. A tick-synchronized parallel beam search over the full game state
//!      (position, velocity, rotation, collision-skip parity, pickup mask),
//!      ranked by a lower-bound heuristic: per-pickup Dijkstra distance
//!      fields + an exact held-karp-style DP over remaining pickup subsets.
//!   3. A local-search polisher that mutates the action tape (delete /
//!      insert / overwrite / boundary-shift) and accepts exact replays that
//!      complete the level in fewer ticks.
//!
//! Everything is deterministic given a seed. Time is measured in ticks of
//! 1/60 s, the same unit the ghost sidecars use.

use std::collections::HashMap;
use std::f32::consts::PI;

use rayon::prelude::*;

use crate::real_game::RealSpaceAceGame;

// --- exact physics constants (must match real_physics.rs) ------------------
const DT: f32 = 1.0 / 60.0;
const GRAVITY: f32 = 100.0;
const THRUST_POWER: f32 = 400.0;
const ROTATION_SPEED: f32 = 4.363323;
const PICKUP_RADIUS_SQ: f32 = (36.5 + 10.0) * (36.5 + 10.0);
const COLLISION_GRID_SIZE: f32 = 500.0;

// Exact ship vertices from real_physics.rs (JavaScript shipVerts).
const SHIP_VERTS: [(f32, f32); 10] = [
    (0.0, -36.5),
    (-19.0, 23.5),
    (-24.0, 23.5),
    (-15.675, 13.0),
    (19.0, 23.5),
    (24.0, 23.5),
    (15.675, 13.0),
    (0.0, 67.45),
    (-14.1075, 13.0),
    (14.1075, 13.0),
];
// Collision segments as vertex index pairs, same order as real_physics.rs.
const SHIP_SEGS: [(usize, usize); 5] = [(3, 6), (2, 1), (1, 0), (0, 4), (4, 5)];

// Same action table as mcts.rs / spaceace.strategies.actions.ALL_ACTIONS:
// (rotate_left, rotate_right, thrust)
const ACTIONS: [(bool, bool, bool); 6] = [
    (false, false, false),
    (false, false, true),
    (true, false, false),
    (true, false, true),
    (false, true, false),
    (false, true, true),
];

const UNREACHABLE: i32 = i32::MAX / 4;

// --- compact game state ------------------------------------------------------

/// Full deterministic game state in 24 bytes. `skip` is the collision-skip
/// frame counter from real_physics (only its parity matters, and it only
/// ever increments, so u8 wrapping preserves behavior). `mask` has bit i set
/// when pickup i has been collected.
#[derive(Clone, Copy, Debug)]
pub struct SimState {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub rot: f32,
    pub skip: u8,
    pub mask: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepOutcome {
    Alive,
    Crashed,
    Completed,
}

// --- deterministic tiny RNG (splitmix64) ------------------------------------

#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[inline]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    z = (z ^ (z >> 33)).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    z ^ (z >> 33)
}

// --- solver ------------------------------------------------------------------

pub struct AceSolver {
    // exact collision world
    lines: Vec<[f32; 4]>,
    // dense 500px-cell index over the map, same cell partition as
    // real_collision.rs (floor(coord / 500)). Cells outside the map have no
    // lines, which matches HashMap misses in the original.
    cgrid_min: (i32, i32),
    cgrid_dims: (i32, i32),
    cgrid: Vec<Vec<u32>>,

    pickups: Vec<(f32, f32)>,
    n_pickups: usize,
    full_mask: u32,
    spawn: SimState,
    /// Strict mode: check wall collision every tick instead of honoring the
    /// engine's every-other-frame skip above ~316 px/s. Strict tapes never
    /// overlap a wall, and they replay identically on the real engine (the
    /// engine's checks are a subset of strict's). Non-strict solving may
    /// thread walls on skipped frames — engine-legal, but it looks like
    /// clipping and was ruled out for fair ghosts.
    strict: bool,

    // heuristic: per-pickup Dijkstra distance fields on a 10px grid
    hcell: f32,
    hmin: (f32, f32),
    hdims: (usize, usize), // (rows, cols)
    fields: Vec<Vec<i32>>, // px-scale distances, -1 where unreached
    // rem[mask * n + p] = length of the shortest pickup-to-pickup path that
    // starts at p and visits every pickup in `mask` (p must be in mask).
    rem: Vec<i32>,
    // pair[p * n + q] = geodesic distance from pickup p to pickup q (px).
    pair: Vec<i32>,
    // next_dir[p * n + q] = unit direction of the route's first step when
    // leaving pickup p toward pickup q (steepest descent of q's field at p).
    next_dir: Vec<(f32, f32)>,
}

pub struct BeamParams {
    pub width: usize,
    pub max_ticks: u32,
    pub seed: u64,
    pub quant_pos: f32,
    pub quant_vel: f32,
    pub rot_bins: u32,
    pub lookahead: f32,
    /// Velocity-reward strength: rank = h_now - mix*(h_now - h_stop) + doom.
    /// Values > 1 overweight momentum that makes route progress.
    pub mix: f32,
    /// Divisor turning speed into the projection horizon (t = v / proj_div);
    /// smaller projects further ahead, rewarding velocity more sharply.
    pub proj_div: f32,
    /// Multiplier on the doom penalty. 1.0 for global solves (extinction is
    /// fatal there). Corridor refinement warm-starts from a proven tape —
    /// the reference node always survives — so it can afford near-zero doom
    /// and let raw physics prune, which unlocks expert-speed cornering.
    pub doom_scale: f32,
    /// Weight of the turnaround penalty charged when the projection collects
    /// a pickup with velocity misaligned to the next leg. Prevents the beam
    /// from favoring fast-but-overshooting approaches whose redirect cost
    /// only becomes visible dozens of layers after the pickup.
    pub turn_w: f32,
    /// Amplitude (in px) of seeded rank noise — decorrelates ties across
    /// seeds so restarts explore different regions of the search space.
    pub jitter: f32,
}

impl Default for BeamParams {
    fn default() -> Self {
        BeamParams {
            width: 50_000,
            max_ticks: 4000,
            seed: 0,
            quant_pos: 6.0,
            quant_vel: 12.0,
            rot_bins: 64,
            lookahead: 1.0,
            mix: 0.8,
            proj_div: 700.0,
            doom_scale: 1.0,
            turn_w: 1.0,
            jitter: 3.0,
        }
    }
}

struct Cand {
    key: u64,
    rank: f32,
    state: SimState,
    parent: u32,
    action: u8,
}

/// Target state for rendezvous prefix re-solves.
struct RendezvousTarget {
    state: SimState,
    tol_pos: f32,
    tol_vel: f32,
    tol_rot: f32,
}

impl RendezvousTarget {
    #[inline]
    fn matches(&self, s: &SimState) -> bool {
        s.mask == self.state.mask
            && (s.x - self.state.x).abs() <= self.tol_pos
            && (s.y - self.state.y).abs() <= self.tol_pos
            && (s.vx - self.state.vx).abs() <= self.tol_vel
            && (s.vy - self.state.vy).abs() <= self.tol_vel
            && {
                let two_pi = 2.0 * PI;
                let mut dr = (s.rot - self.state.rot).rem_euclid(two_pi);
                if dr > PI {
                    dr = two_pi - dr;
                }
                dr <= self.tol_rot
            }
    }
}

/// Dilated occupancy grid around a reference trajectory (see `refine`).
struct Corridor {
    cell: f32,
    rows: usize,
    cols: usize,
    min: (f32, f32),
    occ: Vec<bool>,
}

impl Corridor {
    #[inline]
    fn contains(&self, x: f32, y: f32) -> bool {
        let c = ((x - self.min.0) / self.cell) as i64;
        let r = ((y - self.min.1) / self.cell) as i64;
        if r < 0 || c < 0 || r >= self.rows as i64 || c >= self.cols as i64 {
            return false;
        }
        self.occ[r as usize * self.cols + c as usize]
    }
}

impl AceSolver {
    pub fn from_level(level: usize, strict: bool) -> Result<Self, String> {
        let mut game = RealSpaceAceGame::new();
        game.load_level(level)?;
        Ok(Self::from_game(&game, strict))
    }

    pub fn from_game(game: &RealSpaceAceGame, strict: bool) -> Self {
        let lines: Vec<[f32; 4]> = game
            .get_map_lines()
            .iter()
            .map(|&(x1, y1, x2, y2)| [x1, y1, x2, y2])
            .collect();
        let pickups: Vec<(f32, f32)> = game
            .get_pickup_positions()
            .iter()
            .map(|&(x, y, _)| (x, y))
            .collect();
        let n_pickups = pickups.len();
        assert!(n_pickups <= 20, "solver supports at most 20 pickups (REM DP)");

        // Spawn state: the game must be freshly reset.
        let mut g = game.clone();
        g.reset();
        let st = g.get_state();
        let spawn = SimState {
            x: st.ship_x,
            y: st.ship_y,
            vx: st.ship_vx,
            vy: st.ship_vy,
            rot: st.ship_rotation,
            skip: 0,
            mask: 0,
        };

        // Dense collision-line index over 500px cells.
        let mut min_gx = i32::MAX;
        let mut max_gx = i32::MIN;
        let mut min_gy = i32::MAX;
        let mut max_gy = i32::MIN;
        for l in &lines {
            for &(x, y) in &[(l[0], l[1]), (l[2], l[3])] {
                min_gx = min_gx.min((x / COLLISION_GRID_SIZE).floor() as i32);
                max_gx = max_gx.max((x / COLLISION_GRID_SIZE).floor() as i32);
                min_gy = min_gy.min((y / COLLISION_GRID_SIZE).floor() as i32);
                max_gy = max_gy.max((y / COLLISION_GRID_SIZE).floor() as i32);
            }
        }
        let dims = (max_gx - min_gx + 1, max_gy - min_gy + 1);
        let mut cgrid = vec![Vec::new(); (dims.0 * dims.1) as usize];
        for (i, l) in lines.iter().enumerate() {
            let gx0 = (l[0].min(l[2]) / COLLISION_GRID_SIZE).floor() as i32;
            let gx1 = (l[0].max(l[2]) / COLLISION_GRID_SIZE).floor() as i32;
            let gy0 = (l[1].min(l[3]) / COLLISION_GRID_SIZE).floor() as i32;
            let gy1 = (l[1].max(l[3]) / COLLISION_GRID_SIZE).floor() as i32;
            for gx in gx0..=gx1 {
                for gy in gy0..=gy1 {
                    cgrid[((gx - min_gx) * dims.1 + (gy - min_gy)) as usize].push(i as u32);
                }
            }
        }

        // Heuristic distance fields.
        let bounds = game.get_map_bounds();
        let hcell = 10.0f32;
        let cols = ((bounds.max_x - bounds.min_x) / hcell) as usize + 1;
        let rows = ((bounds.max_y - bounds.min_y) / hcell) as usize + 1;
        let hmin = (bounds.min_x, bounds.min_y);

        // Blocked grids at progressively tighter inflation. A pickup's field
        // uses the widest inflation that still reaches the spawn cell, so
        // narrow-corridor pickups automatically fall back to tighter grids.
        //
        // 30px inflation (< ship wingspan 48px) is mildly pessimistic; that
        // is intentional. Tighter inflations were tried and lure the beam
        // deep into converging dead-ends (e.g. the L7 tower notch), where
        // whole cohorts die; the loss outweighs the better route estimates.
        let inflations = [30.0f32, 20.0, 12.0];
        let blocked_grids: Vec<Vec<bool>> = inflations
            .iter()
            .map(|&inf| build_blocked(rows, cols, hmin, hcell, &lines, inf))
            .collect();

        let spawn_cell = cell_of(rows, cols, hmin, hcell, spawn.x, spawn.y);
        let fields: Vec<Vec<i32>> = pickups
            .par_iter()
            .map(|&(px, py)| {
                for blocked in &blocked_grids {
                    let f = dijkstra(rows, cols, hmin, hcell, blocked, px, py);
                    let reach = spiral_read(&f, rows, cols, spawn_cell, 4);
                    if reach < UNREACHABLE {
                        return f;
                    }
                }
                // Last resort: tightest grid even if the spawn lookup failed;
                // spiral reads at query time may still find values.
                dijkstra(rows, cols, hmin, hcell, blocked_grids.last().unwrap(), px, py)
            })
            .collect();

        // Pairwise pickup distances (read pickup i's position in pickup j's field)
        // and the outgoing route direction at each pickup toward each other
        // pickup (downhill gradient of j's field around i's position).
        let n = n_pickups;
        let mut pair = vec![0i32; n * n];
        let mut next_dir = vec![(0.0f32, 0.0f32); n * n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let c = cell_of(rows, cols, hmin, hcell, pickups[i].0, pickups[i].1);
                pair[i * n + j] = spiral_read(&fields[j], rows, cols, c, 6);
                // Central differences at ±3 cells; fall back to the straight
                // line between the pickups if the field is flat/unreadable.
                let probe = 3.0 * hcell;
                let (px, py) = pickups[i];
                let dxp = spiral_read(&fields[j], rows, cols, cell_of(rows, cols, hmin, hcell, px + probe, py), 3);
                let dxm = spiral_read(&fields[j], rows, cols, cell_of(rows, cols, hmin, hcell, px - probe, py), 3);
                let dyp = spiral_read(&fields[j], rows, cols, cell_of(rows, cols, hmin, hcell, px, py + probe), 3);
                let dym = spiral_read(&fields[j], rows, cols, cell_of(rows, cols, hmin, hcell, px, py - probe), 3);
                let mut gx = -((dxp - dxm) as f32);
                let mut gy = -((dyp - dym) as f32);
                let norm = (gx * gx + gy * gy).sqrt();
                if norm < 1e-3 || dxp >= UNREACHABLE || dxm >= UNREACHABLE || dyp >= UNREACHABLE || dym >= UNREACHABLE {
                    gx = pickups[j].0 - px;
                    gy = pickups[j].1 - py;
                    let d = (gx * gx + gy * gy).sqrt().max(1e-3);
                    next_dir[i * n + j] = (gx / d, gy / d);
                } else {
                    next_dir[i * n + j] = (gx / norm, gy / norm);
                }
            }
        }

        // REM DP over subsets: rem[mask][p] = shortest path starting at p
        // visiting all pickups in mask (p ∈ mask).
        let mut rem = vec![UNREACHABLE; (1usize << n) * n.max(1)];
        for p in 0..n {
            rem[(1usize << p) * n + p] = 0;
        }
        for mask in 1..(1usize << n) {
            if mask.count_ones() < 2 {
                continue;
            }
            for p in 0..n {
                if mask & (1 << p) == 0 {
                    continue;
                }
                let rest = mask ^ (1 << p);
                let mut best = UNREACHABLE;
                for q in 0..n {
                    if rest & (1 << q) == 0 {
                        continue;
                    }
                    let v = pair[p * n + q].saturating_add(rem[rest * n + q]);
                    if v < best {
                        best = v;
                    }
                }
                rem[mask * n + p] = best;
            }
        }

        AceSolver {
            lines,
            cgrid_min: (min_gx, min_gy),
            cgrid_dims: dims,
            cgrid,
            strict,
            pickups,
            n_pickups,
            full_mask: if n_pickups == 32 { u32::MAX } else { (1u32 << n_pickups) - 1 },
            spawn,
            hcell,
            hmin,
            hdims: (rows, cols),
            fields,
            rem,
            pair,
            next_dir,
        }
    }

    pub fn spawn(&self) -> SimState {
        self.spawn
    }

    pub fn n_pickups(&self) -> usize {
        self.n_pickups
    }

    // --- exact stepping ------------------------------------------------------

    /// Advance one tick with action `a`. Mirrors RealSpaceAceGame::step with
    /// dt = 1/60 exactly (same constants, same operation order, all f32).
    #[inline]
    pub fn step(&self, s: &mut SimState, a: u8) -> StepOutcome {
        let (left, right, thrust) = ACTIONS[a as usize];

        // real_physics::update
        if left {
            s.rot -= ROTATION_SPEED * DT;
        }
        if right {
            s.rot += ROTATION_SPEED * DT;
        }
        if thrust {
            let angle = s.rot - PI * 0.5;
            s.vx += THRUST_POWER * angle.cos() * DT;
            s.vy += THRUST_POWER * angle.sin() * DT;
        }
        s.vy += GRAVITY * DT;
        s.x += s.vx * DT;
        s.y += s.vy * DT;

        // real_physics::should_skip_collision (parity always tracked so the
        // state stays engine-identical; strict mode just refuses to use it)
        let speed_sq = s.vx * s.vx + s.vy * s.vy;
        let skip_collision = if speed_sq > 100000.0 {
            s.skip = s.skip.wrapping_add(1);
            !self.strict && s.skip % 2 != 0
        } else {
            false
        };

        if !skip_collision && self.ship_hits_wall(s.x, s.y, s.rot) {
            return StepOutcome::Crashed;
        }

        // pickup collection (real_physics::check_pickup_collision)
        let mut remaining = self.full_mask & !s.mask;
        while remaining != 0 {
            let p = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            let (px, py) = self.pickups[p];
            let dx = s.x - px;
            let dy = s.y - py;
            if dx * dx + dy * dy <= PICKUP_RADIUS_SQ {
                s.mask |= 1 << p;
            }
        }

        if s.mask == self.full_mask {
            StepOutcome::Completed
        } else {
            StepOutcome::Alive
        }
    }

    #[inline]
    fn ship_hits_wall(&self, x: f32, y: f32, rot: f32) -> bool {
        // Transform the 10 ship verts (same math as real_physics).
        let cos_r = rot.cos();
        let sin_r = rot.sin();
        let mut tv = [(0.0f32, 0.0f32); 10];
        for (i, &(vx, vy)) in SHIP_VERTS.iter().enumerate() {
            tv[i] = (vx * cos_r - vy * sin_r + x, vx * sin_r + vy * cos_r + y);
        }
        for &(a, b) in &SHIP_SEGS {
            let (sx1, sy1) = tv[a];
            let (sx2, sy2) = tv[b];
            let gx0 = (sx1.min(sx2) / COLLISION_GRID_SIZE).floor() as i32;
            let gx1 = (sx1.max(sx2) / COLLISION_GRID_SIZE).floor() as i32;
            let gy0 = (sy1.min(sy2) / COLLISION_GRID_SIZE).floor() as i32;
            let gy1 = (sy1.max(sy2) / COLLISION_GRID_SIZE).floor() as i32;
            for gx in gx0..=gx1 {
                for gy in gy0..=gy1 {
                    let cx = gx - self.cgrid_min.0;
                    let cy = gy - self.cgrid_min.1;
                    if cx < 0 || cy < 0 || cx >= self.cgrid_dims.0 || cy >= self.cgrid_dims.1 {
                        continue;
                    }
                    for &li in &self.cgrid[(cx * self.cgrid_dims.1 + cy) as usize] {
                        let l = &self.lines[li as usize];
                        if segments_intersect(sx1, sy1, sx2, sy2, l[0], l[1], l[2], l[3]) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Replay an action tape from the spawn. Returns (completed, crashed,
    /// ticks) where ticks is the completion tick if completed, else the
    /// number of ticks survived.
    pub fn replay(&self, tape: &[u8]) -> (bool, bool, u32) {
        let mut s = self.spawn;
        for (i, &a) in tape.iter().enumerate() {
            match self.step(&mut s, a) {
                StepOutcome::Completed => return (true, false, i as u32 + 1),
                StepOutcome::Crashed => return (false, true, i as u32 + 1),
                StepOutcome::Alive => {}
            }
        }
        (false, false, tape.len() as u32)
    }

    /// Replay a prefix and return the state, or None if it dies/completes early.
    pub fn state_after(&self, tape: &[u8]) -> Option<SimState> {
        let mut s = self.spawn;
        for &a in tape {
            if self.step(&mut s, a) != StepOutcome::Alive {
                return None;
            }
        }
        Some(s)
    }

    // --- debug probes ----------------------------------------------------------

    pub fn h_at(&self, x: f32, y: f32, mask: u32) -> i64 {
        self.h_px(x, y, mask) as i64
    }

    pub fn trace(&self, tape: &[u8], stride: usize) -> Vec<(u32, f32, f32, i64)> {
        let mut s = self.spawn;
        let mut out = vec![(0, s.x, s.y, self.h_px(s.x, s.y, s.mask) as i64)];
        for (i, &a) in tape.iter().enumerate() {
            if self.step(&mut s, a) != StepOutcome::Alive {
                break;
            }
            if (i + 1) % stride.max(1) == 0 {
                out.push((i as u32 + 1, s.x, s.y, self.h_px(s.x, s.y, s.mask) as i64));
            }
        }
        out
    }

    // --- heuristic -----------------------------------------------------------

    /// Lower-bound style estimate (in px of geodesic travel) of the distance
    /// still to fly: nearest remaining pickup through its distance field plus
    /// the optimal tour over the rest (exact subset DP).
    #[inline]
    fn h_px(&self, x: f32, y: f32, mask: u32) -> i32 {
        let remaining = (self.full_mask & !mask) as usize;
        if remaining == 0 {
            return 0;
        }
        let (rows, cols) = self.hdims;
        let cell = cell_of(rows, cols, self.hmin, self.hcell, x, y);
        let n = self.n_pickups;
        let mut best = UNREACHABLE;
        let mut bits = remaining;
        while bits != 0 {
            let p = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let d = spiral_read(&self.fields[p], rows, cols, cell, 2);
            let v = d.saturating_add(self.rem[remaining * n + p]);
            if v < best {
                best = v;
            }
        }
        best
    }

    // --- beam search -----------------------------------------------------------

    /// Tick-synchronized beam search from `start`. Returns the shortest action
    /// tape (within the beam) that completes the level, or None.
    pub fn beam_from(&self, start: SimState, p: &BeamParams) -> Option<Vec<u8>> {
        self.beam_from_corridor(start, p, None)
    }

    fn beam_from_corridor(
        &self,
        start: SimState,
        p: &BeamParams,
        corridor: Option<&Corridor>,
    ) -> Option<Vec<u8>> {
        self.beam_impl(start, p, corridor, None)
    }

    /// Core beam loop. `warm` optionally carries a reference trajectory
    /// (states after each tick, and the action taken at each tick) that is
    /// force-injected into every layer: the beam then cannot do worse than
    /// the reference, and any locally faster deviation strictly improves it.
    fn beam_impl(
        &self,
        start: SimState,
        p: &BeamParams,
        corridor: Option<&Corridor>,
        warm: Option<(&[SimState], &[u8])>,
    ) -> Option<Vec<u8>> {
        if start.mask == self.full_mask {
            return Some(Vec::new());
        }
        let mut cur: Vec<SimState> = vec![start];
        let mut links: Vec<(Vec<u32>, Vec<u8>)> = Vec::new();
        let debug = std::env::var("ACE_DEBUG").is_ok();
        // Index of the reference node inside `cur` while warm-starting.
        let mut ref_idx: Option<u32> = warm.map(|_| 0);

        for tick in 0..p.max_ticks {
            // Expand all nodes in parallel.
            let chunk = (cur.len() / (rayon::current_num_threads() * 4)).max(64);
            let cands: Vec<Cand> = cur
                .par_chunks(chunk)
                .enumerate()
                .flat_map_iter(|(ci, states)| {
                    let base = ci * chunk;
                    let mut out = Vec::with_capacity(states.len() * 6);
                    for (si, s0) in states.iter().enumerate() {
                        for a in 0..6u8 {
                            let mut s = *s0;
                            match self.step(&mut s, a) {
                                StepOutcome::Crashed => continue,
                                _ => {}
                            }
                            if s.mask != self.full_mask {
                                if let Some(c) = corridor {
                                    if !c.contains(s.x, s.y) {
                                        continue;
                                    }
                                }
                            }
                            let key = self.quant_key(&s, p);
                            let rank = self.rank(&s, p, key);
                            out.push(Cand {
                                key,
                                rank,
                                state: s,
                                parent: (base + si) as u32,
                                action: a,
                            });
                        }
                    }
                    out
                })
                .collect();

            // Completion check: all candidates are at the same tick, so the
            // first completing layer is beam-optimal; pick any completer.
            if let Some(win) = cands.iter().find(|c| c.state.mask == self.full_mask) {
                let mut tape = vec![win.action];
                let mut parent = win.parent;
                for (parents, actions) in links.iter().rev() {
                    tape.push(actions[parent as usize]);
                    parent = parents[parent as usize];
                }
                tape.reverse();
                return Some(tape);
            }

            // Dedup by quantized state, keeping the best-ranked candidate.
            let mut seen: HashMap<u64, usize> = HashMap::with_capacity(cands.len());
            let mut keep: Vec<usize> = Vec::with_capacity(cands.len());
            for (i, c) in cands.iter().enumerate() {
                match seen.entry(c.key) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(keep.len());
                        keep.push(i);
                    }
                    std::collections::hash_map::Entry::Occupied(e) => {
                        let slot = *e.get();
                        if c.rank < cands[keep[slot]].rank {
                            keep[slot] = i;
                        }
                    }
                }
            }

            // Select the beam, stratified by pickup mask: each distinct
            // collected-set gets an equal share of the width so one
            // fast-moving cluster can't extinguish alternative routes or
            // more cautious pacing (which is how beams die en masse).
            if keep.len() > p.width {
                // BTreeMap: deterministic group order (HashMap iteration
                // order varies per process, breaking run reproducibility).
                let mut by_mask: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
                for &i in &keep {
                    by_mask.entry(cands[i].state.mask).or_default().push(i);
                }
                // Drop route-orderings that have fallen hopelessly behind:
                // at equal tick, rank ≈ remaining route px, so a group whose
                // best is far worse than the global best is a strictly worse
                // ordering and would otherwise hold its quota forever.
                if by_mask.len() > 2 {
                    let global_best = keep
                        .iter()
                        .map(|&i| cands[i].rank)
                        .fold(f32::INFINITY, f32::min);
                    by_mask.retain(|_, group| {
                        group
                            .iter()
                            .map(|&i| cands[i].rank)
                            .fold(f32::INFINITY, f32::min)
                            <= global_best + 900.0
                    });
                }
                let quota = (p.width / by_mask.len().max(1)).max(256);
                let mut selected: Vec<usize> = Vec::with_capacity(p.width + quota);
                let mut leftover: Vec<usize> = Vec::new();
                for (_, mut group) in by_mask {
                    if group.len() > quota {
                        group.select_nth_unstable_by(quota, |&a, &b| {
                            cands[a].rank.partial_cmp(&cands[b].rank).unwrap()
                        });
                        leftover.extend_from_slice(&group[quota..]);
                        group.truncate(quota);
                    }
                    selected.extend_from_slice(&group);
                }
                // Fill remaining width with the best leftovers regardless of mask.
                if selected.len() < p.width && !leftover.is_empty() {
                    let fill = (p.width - selected.len()).min(leftover.len());
                    if leftover.len() > fill {
                        leftover.select_nth_unstable_by(fill, |&a, &b| {
                            cands[a].rank.partial_cmp(&cands[b].rank).unwrap()
                        });
                        leftover.truncate(fill);
                    }
                    selected.extend_from_slice(&leftover);
                }
                keep = selected;
            }
            if debug && (tick % 60 == 0 || keep.is_empty()) {
                let mut best_h = i32::MAX;
                let mut best_pos = (0.0f32, 0.0f32, 0u32);
                let mut doomed = 0usize;
                let mut speed_sum = 0.0f64;
                for &i in &keep {
                    let s = &cands[i].state;
                    let h = self.h_px(s.x, s.y, s.mask);
                    if h < best_h {
                        best_h = h;
                        best_pos = (s.x, s.y, s.mask);
                    }
                    let speed = (s.vx * s.vx + s.vy * s.vy).sqrt();
                    speed_sum += speed as f64;
                    if self.doom_penalty(s) > 0.0 {
                        doomed += 1;
                    }
                }
                let max_bits = keep.iter().map(|&i| cands[i].state.mask.count_ones()).max().unwrap_or(0);
                eprintln!(
                    "[beam] tick={} cands={} kept={} best_h={} at ({:.0},{:.0}) mask={:b} max_pickups={} doomed={}/{} mean_speed={:.0}",
                    tick, cands.len(), keep.len(), best_h, best_pos.0, best_pos.1, best_pos.2, max_bits, doomed, keep.len(),
                    if keep.is_empty() { 0.0 } else { speed_sum / keep.len() as f64 }
                );
            }
            if keep.is_empty() && ref_idx.is_none() {
                return None; // every branch crashed
            }

            let mut parents = Vec::with_capacity(keep.len() + 1);
            let mut actions = Vec::with_capacity(keep.len() + 1);
            let mut next = Vec::with_capacity(keep.len() + 1);
            for &i in &keep {
                let c = &cands[i];
                parents.push(c.parent);
                actions.push(c.action);
                next.push(c.state);
            }

            // Warm start: force-keep the reference node for the next layer.
            if let (Some((ref_states, ref_actions)), Some(pi)) = (warm, ref_idx) {
                let t1 = tick as usize + 1;
                if t1 < ref_states.len() {
                    parents.push(pi);
                    actions.push(ref_actions[tick as usize]);
                    next.push(ref_states[t1]);
                    ref_idx = Some(next.len() as u32 - 1);
                } else {
                    ref_idx = None;
                }
            }

            links.push((parents, actions));
            cur = next;
        }
        None
    }

    #[inline]
    fn quant_key(&self, s: &SimState, p: &BeamParams) -> u64 {
        let qx = (s.x / p.quant_pos).floor() as i64;
        let qy = (s.y / p.quant_pos).floor() as i64;
        let qvx = (s.vx / p.quant_vel).floor() as i64;
        let qvy = (s.vy / p.quant_vel).floor() as i64;
        let two_pi = 2.0 * PI;
        let rot_wrapped = s.rot.rem_euclid(two_pi);
        let qr = ((rot_wrapped / two_pi) * p.rot_bins as f32) as i64 % p.rot_bins as i64;
        let mut k = mix64(qx as u64 ^ 0x517c_c1b7_2722_0a95);
        k = mix64(k ^ qy as u64);
        k = mix64(k ^ qvx as u64);
        k = mix64(k ^ qvy as u64);
        k = mix64(k ^ qr as u64);
        // Collision-skip parity is real state: at speed the engine only
        // checks walls every other tick, so two otherwise-identical states
        // with different parity can thread different corner clips.
        k = mix64(k ^ s.mask as u64 ^ ((s.skip as u64 & 1) << 63));
        k
    }

    /// Distance to the nearest wall along a ray (heuristic use only, so the
    /// math need not match the engine). Tests all lines in the 500px cells
    /// the ray's bbox touches.
    fn ray_wall_dist(&self, x: f32, y: f32, ux: f32, uy: f32, max_d: f32) -> f32 {
        let ex = x + ux * max_d;
        let ey = y + uy * max_d;
        let gx0 = (x.min(ex) / COLLISION_GRID_SIZE).floor() as i32;
        let gx1 = (x.max(ex) / COLLISION_GRID_SIZE).floor() as i32;
        let gy0 = (y.min(ey) / COLLISION_GRID_SIZE).floor() as i32;
        let gy1 = (y.max(ey) / COLLISION_GRID_SIZE).floor() as i32;
        let mut best = max_d;
        for gx in gx0..=gx1 {
            for gy in gy0..=gy1 {
                let cx = gx - self.cgrid_min.0;
                let cy = gy - self.cgrid_min.1;
                if cx < 0 || cy < 0 || cx >= self.cgrid_dims.0 || cy >= self.cgrid_dims.1 {
                    continue;
                }
                for &li in &self.cgrid[(cx * self.cgrid_dims.1 + cy) as usize] {
                    let l = &self.lines[li as usize];
                    let s2x = l[2] - l[0];
                    let s2y = l[3] - l[1];
                    let s1x = ex - x;
                    let s1y = ey - y;
                    let denom = -s2x * s1y + s1x * s2y;
                    if denom.abs() < 1e-6 {
                        continue;
                    }
                    let s = (-s1y * (x - l[0]) + s1x * (y - l[1])) / denom;
                    let t = (s2x * (y - l[1]) - s2y * (x - l[0])) / denom;
                    if s >= 0.0 && s <= 1.0 && t >= 0.0 && t <= 1.0 {
                        let d = t * max_d;
                        if d < best {
                            best = d;
                        }
                    }
                }
            }
        }
        best
    }

    /// Beam rank: h at the current position blended with h at the braking
    /// point (rewards useful momentum), plus a hard doom penalty when the
    /// ship physically cannot avoid the wall ahead: stopping distance
    /// includes the time to rotate to retrograde (4.36 rad/s) and the
    /// v^2/2a braking run, compared against a raycast along the velocity.
    /// Doomed states must never displace viable ones — a final full-speed
    /// dash into the last pickup still works because completion is detected
    /// during expansion, before ranking matters. Small seeded jitter
    /// decorrelates equal-ranked states across seeds.
    #[inline]
    fn doom_penalty(&self, s: &SimState) -> f32 {
        let speed = (s.vx * s.vx + s.vy * s.vy).sqrt();
        if speed <= 60.0 {
            return 0.0;
        }
        let ux = s.vx / speed;
        let uy = s.vy / speed;
        // Rotation needed to point the thrust vector against velocity.
        let rot_target = (-s.vy).atan2(-s.vx) + PI * 0.5;
        let mut dth = (s.rot - rot_target).rem_euclid(2.0 * PI);
        if dth > PI {
            dth = 2.0 * PI - dth;
        }
        let t_rot = dth / ROTATION_SPEED;
        // Direction-aware braking: gravity helps kill upward velocity
        // (uy < 0 → decel ≈ 500) and fights the brake on descents
        // (uy > 0 → decel ≈ 300). A fixed conservative value over-brakes
        // climbs, which is where expert lines carry the most speed.
        let a_eff = (400.0 - 100.0 * uy).clamp(280.0, 490.0);
        // Ship nose sticks out ~40px.
        let d_need = speed * t_rot + speed * speed / (2.0 * a_eff) + 40.0;
        let d_ahead = self.ray_wall_dist(s.x, s.y, ux, uy, d_need);
        if d_ahead >= d_need {
            return 0.0;
        }
        3000.0 + (d_need - d_ahead)
    }

    #[inline]
    fn rank(&self, s: &SimState, p: &BeamParams, key: u64) -> f32 {
        let h_now = self.h_px(s.x, s.y, s.mask) as f32;
        let speed = (s.vx * s.vx + s.vy * s.vy).sqrt();
        let mut t_stop = (speed / p.proj_div).min(p.lookahead);
        // Line-of-sight clamp: never project the velocity reward through a
        // wall. With thin partitions, the far side can be much closer to the
        // goal, and an unclamped projection would reward flying into walls.
        if speed > 60.0 {
            let proj_dist = speed * t_stop;
            let d_ahead = self.ray_wall_dist(s.x, s.y, s.vx / speed, s.vy / speed, proj_dist);
            if d_ahead < proj_dist {
                t_stop = (d_ahead - 20.0).max(0.0) / speed;
            }
        }
        let ex = s.x + s.vx * t_stop;
        let ey = s.y + s.vy * t_stop;

        // Fly-through credit: if the projected segment passes through a
        // remaining pickup's collection radius, evaluate the projection with
        // that pickup collected. Without this, h measures distance TO the
        // pickup, so a fast fly-through looks like overshoot and the rank
        // rewards braking to a stop at every pickup — the single biggest
        // gap between "AI that completes the level" and expert racing lines.
        let mut proj_mask = s.mask;
        let mut rem = self.full_mask & !s.mask;
        while rem != 0 {
            let pi = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            let (px, py) = self.pickups[pi];
            // 40px: collection radius (46.5) minus a small aiming margin.
            if point_seg_dist_sq(px, py, s.x, s.y, ex, ey) <= 40.0 * 40.0 {
                proj_mask |= 1 << pi;
            }
        }

        // Final-dash exemption: if the projection completes the level, the
        // ship doesn't need to survive afterwards — no doom, and h_stop is 0.
        let doom = if proj_mask == self.full_mask {
            0.0
        } else {
            self.doom_penalty(s) * p.doom_scale
        };
        let h_stop = self.h_px(ex, ey, proj_mask) as f32;
        let h_stop = if h_stop >= UNREACHABLE as f32 { h_now } else { h_stop };

        // Turnaround charge: arriving at a pickup with velocity that doesn't
        // point down the next leg costs v_wasted^2 / 2a to redirect. Without
        // this, the beam picks whoever reaches the pickup first — usually a
        // fast overshooting approach that then scrambles back.
        let mut turn_pen = 0.0f32;
        let collected_now = proj_mask & !s.mask;
        if collected_now != 0 && proj_mask != self.full_mask && p.turn_w > 0.0 {
            let n = self.n_pickups;
            let pi = collected_now.trailing_zeros() as usize;
            let remaining = (self.full_mask & !proj_mask) as usize;
            let mut best_cost = i32::MAX;
            let mut best_q = usize::MAX;
            let mut bits = remaining;
            while bits != 0 {
                let q = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let cost = self.pair[pi * n + q].saturating_add(self.rem[remaining * n + q]);
                if cost < best_cost {
                    best_cost = cost;
                    best_q = q;
                }
            }
            if best_q != usize::MAX {
                let (dx, dy) = self.next_dir[pi * n + best_q];
                let vd = s.vx * dx + s.vy * dy;
                let waste_sq = if vd <= 0.0 {
                    speed * speed
                } else {
                    (speed * speed - vd * vd).max(0.0)
                };
                turn_pen = p.turn_w * waste_sq / 700.0;
            }
        }

        let jitter = if p.jitter > 0.0 {
            (mix64(key ^ p.seed) & 0xFFFF) as f32 / 65535.0 * p.jitter
        } else {
            0.0
        };
        (1.0 - p.mix) * h_now + p.mix * h_stop + doom + turn_pen + jitter
    }

    /// Full solve from spawn.
    pub fn solve(&self, p: &BeamParams) -> Option<Vec<u8>> {
        self.beam_from_corridor(self.spawn, p, None)
    }

    /// Corridor refinement: re-search restricted to a tube of `radius` px
    /// around the reference tape's trajectory (time-free, path-space).
    /// Use finer quantization than the global solve — the beam width is
    /// concentrated inside the tube, so it explores micro-optimizations of
    /// the racing line the coarse global search can't represent. Returns a
    /// strictly shorter tape or None.
    pub fn refine(&self, tape: &[u8], radius: f32, p: &BeamParams) -> Option<Vec<u8>> {
        let (completed, _, ticks) = self.replay(tape);
        if !completed {
            return None;
        }
        // Occupancy grid over the reference path, dilated by `radius`.
        let cell = 25.0f32;
        let cols = ((self.hdims.1 as f32 * self.hcell) / cell) as usize + 2;
        let rows = ((self.hdims.0 as f32 * self.hcell) / cell) as usize + 2;
        let mut occ = vec![false; rows * cols];
        let rad_cells = (radius / cell).ceil() as i32;
        let mut s = self.spawn;
        let mut mark = |x: f32, y: f32| {
            let c0 = ((x - self.hmin.0) / cell) as i32;
            let r0 = ((y - self.hmin.1) / cell) as i32;
            for dr in -rad_cells..=rad_cells {
                for dc in -rad_cells..=rad_cells {
                    let r = r0 + dr;
                    let c = c0 + dc;
                    if r < 0 || c < 0 || r >= rows as i32 || c >= cols as i32 {
                        continue;
                    }
                    let dx = dc as f32 * cell;
                    let dy = dr as f32 * cell;
                    if dx * dx + dy * dy <= (radius + cell) * (radius + cell) {
                        occ[r as usize * cols + c as usize] = true;
                    }
                }
            }
        };
        mark(s.x, s.y);
        for &a in &tape[..ticks as usize] {
            if self.step(&mut s, a) != StepOutcome::Alive {
                break;
            }
            mark(s.x, s.y);
        }
        let corridor = Corridor { cell, rows, cols, min: self.hmin, occ };

        // Reference trajectory for warm-starting: state after each tick.
        let mut ref_states = Vec::with_capacity(ticks as usize + 1);
        let mut st = self.spawn;
        ref_states.push(st);
        for &a in &tape[..ticks as usize] {
            if self.step(&mut st, a) != StepOutcome::Alive {
                break; // final (completing) state is intentionally excluded
            }
            ref_states.push(st);
        }

        let params = BeamParams { max_ticks: ticks - 1, ..*p };
        let out = self.beam_impl(
            self.spawn,
            &params,
            Some(&corridor),
            Some((&ref_states, &tape[..ticks as usize])),
        )?;
        if (out.len() as u32) < ticks {
            Some(out)
        } else {
            None
        }
    }

    /// Rendezvous prefix re-solve: beam-search a shorter path from spawn to
    /// a state matching the tape's state at `rendezvous_tick` (within the
    /// given tolerances, same pickup mask), then splice the original suffix
    /// and validate by exact replay. This attacks the *beginning* of a tape,
    /// which suffix re-solves and warm-started refinement structurally
    /// cannot improve (early deviations must otherwise re-earn the entire
    /// remaining route before they are adopted).
    pub fn resolve_prefix(
        &self,
        tape: &[u8],
        rendezvous_tick: usize,
        tol_pos: f32,
        tol_vel: f32,
        tol_rot: f32,
        p: &BeamParams,
    ) -> Option<Vec<u8>> {
        let (completed, _, total) = self.replay(tape);
        if !completed || rendezvous_tick + 8 >= total as usize {
            return None;
        }
        let target = self.state_after(&tape[..rendezvous_tick])?;
        let suffix = &tape[rendezvous_tick..total as usize];

        // Reference states for warm-starting the prefix search.
        let mut ref_states = Vec::with_capacity(rendezvous_tick + 1);
        let mut st = self.spawn;
        ref_states.push(st);
        for &a in &tape[..rendezvous_tick] {
            if self.step(&mut st, a) != StepOutcome::Alive {
                return None;
            }
            ref_states.push(st);
        }

        let params = BeamParams { max_ticks: rendezvous_tick as u32 - 1, ..*p };
        let matcher = RendezvousTarget { state: target, tol_pos, tol_vel, tol_rot };
        let candidates = self.beam_prefix_matches(
            self.spawn, &params, &matcher,
            (&ref_states, &tape[..rendezvous_tick]),
        );
        // Earliest-arriving candidates first: each saves (rendezvous_tick - t).
        let debug = std::env::var("ACE_DEBUG").is_ok();
        if debug {
            eprintln!(
                "[rendezvous] target tick {} -> {} match candidates (earliest {:?})",
                rendezvous_tick,
                candidates.len(),
                candidates.first().map(|c| c.0)
            );
        }
        for (arrive_tick, mut prefix) in candidates {
            prefix.extend_from_slice(suffix);
            let (ok, _, ticks) = self.replay(&prefix);
            if debug {
                eprintln!(
                    "[rendezvous] splice arrive={} (saves {}): completed={} ticks={}",
                    arrive_tick, rendezvous_tick as i64 - arrive_tick as i64, ok, ticks
                );
            }
            if ok && ticks < total {
                prefix.truncate(ticks as usize);
                return Some(prefix);
            }
        }
        None
    }

    /// Beam over the prefix window collecting states that match `matcher`
    /// earlier than the reference. Returns reconstructed prefixes sorted by
    /// arrival tick (earliest first).
    fn beam_prefix_matches(
        &self,
        start: SimState,
        p: &BeamParams,
        matcher: &RendezvousTarget,
        warm: (&[SimState], &[u8]),
    ) -> Vec<(u32, Vec<u8>)> {
        let mut cur: Vec<SimState> = vec![start];
        let mut links: Vec<(Vec<u32>, Vec<u8>)> = Vec::new();
        let mut ref_idx: Option<u32> = Some(0);
        let mut found: Vec<(u32, Vec<u8>)> = Vec::new();

        for tick in 0..p.max_ticks {
            let chunk = (cur.len() / (rayon::current_num_threads() * 4)).max(64);
            let cands: Vec<Cand> = cur
                .par_chunks(chunk)
                .enumerate()
                .flat_map_iter(|(ci, states)| {
                    let base = ci * chunk;
                    let mut out = Vec::with_capacity(states.len() * 6);
                    for (si, s0) in states.iter().enumerate() {
                        for a in 0..6u8 {
                            let mut s = *s0;
                            if self.step(&mut s, a) == StepOutcome::Crashed {
                                continue;
                            }
                            let key = self.quant_key(&s, p);
                            let rank = self.rank_to_target(&s, p, key, matcher);
                            out.push(Cand { key, rank, state: s, parent: (base + si) as u32, action: a });
                        }
                    }
                    out
                })
                .collect();

            // Collect matches at this tick (they still go into the beam too).
            for c in &cands {
                if matcher.matches(&c.state) {
                    let mut prefix = vec![c.action];
                    let mut parent = c.parent;
                    for (parents, actions) in links.iter().rev() {
                        prefix.push(actions[parent as usize]);
                        parent = parents[parent as usize];
                    }
                    prefix.reverse();
                    found.push((tick + 1, prefix));
                }
            }
            if found.len() >= 64 {
                break; // plenty of splice candidates
            }

            let mut seen: HashMap<u64, usize> = HashMap::with_capacity(cands.len());
            let mut keep: Vec<usize> = Vec::with_capacity(cands.len());
            for (i, c) in cands.iter().enumerate() {
                match seen.entry(c.key) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(keep.len());
                        keep.push(i);
                    }
                    std::collections::hash_map::Entry::Occupied(e) => {
                        let slot = *e.get();
                        if c.rank < cands[keep[slot]].rank {
                            keep[slot] = i;
                        }
                    }
                }
            }
            if keep.len() > p.width {
                keep.select_nth_unstable_by(p.width, |&a, &b| {
                    cands[a].rank.partial_cmp(&cands[b].rank).unwrap()
                });
                keep.truncate(p.width);
            }
            if keep.is_empty() && ref_idx.is_none() {
                break;
            }

            let mut parents = Vec::with_capacity(keep.len() + 1);
            let mut actions = Vec::with_capacity(keep.len() + 1);
            let mut next = Vec::with_capacity(keep.len() + 1);
            for &i in &keep {
                let c = &cands[i];
                parents.push(c.parent);
                actions.push(c.action);
                next.push(c.state);
            }
            let (ref_states, ref_actions) = warm;
            if let Some(pi) = ref_idx {
                let t1 = tick as usize + 1;
                if t1 < ref_states.len() {
                    parents.push(pi);
                    actions.push(ref_actions[tick as usize]);
                    next.push(ref_states[t1]);
                    ref_idx = Some(next.len() as u32 - 1);
                } else {
                    ref_idx = None;
                }
            }
            links.push((parents, actions));
            cur = next;
        }
        found.sort_by_key(|&(t, _)| t);
        found
    }

    /// Rank for the rendezvous prefix search: distance to the target state,
    /// with the usual doom/jitter, plus velocity-matching pressure.
    #[inline]
    fn rank_to_target(&self, s: &SimState, p: &BeamParams, key: u64, m: &RendezvousTarget) -> f32 {
        if s.mask != m.state.mask {
            // Wrong pickup set: rank by normal route heuristic (must collect
            // the missing pickups first).
            return self.rank(s, p, key) + 5000.0;
        }
        let dx = s.x - m.state.x;
        let dy = s.y - m.state.y;
        let dvx = s.vx - m.state.vx;
        let dvy = s.vy - m.state.vy;
        let pos_d = (dx * dx + dy * dy).sqrt();
        let vel_d = (dvx * dvx + dvy * dvy).sqrt();
        let doom = self.doom_penalty(s) * p.doom_scale;
        let jitter = (mix64(key ^ p.seed) & 0xFFFF) as f32 / 65535.0 * p.jitter;
        pos_d + 0.5 * vel_d + doom + jitter
    }

    /// Re-solve the tape suffix from `from_tick` with a beam bounded to beat
    /// the current tape. Returns a strictly shorter full tape on success.
    pub fn resolve_suffix(&self, tape: &[u8], from_tick: usize, p: &BeamParams) -> Option<Vec<u8>> {
        let (completed, _, total) = self.replay(tape);
        if !completed || from_tick >= total as usize {
            return None;
        }
        let prefix = &tape[..from_tick];
        let start = self.state_after(prefix)?;
        let budget = (total as usize - from_tick).saturating_sub(1);
        if budget == 0 {
            return None;
        }
        let mut params = BeamParams { max_ticks: budget as u32, ..*p };
        params.seed = p.seed;
        let suffix = self.beam_from(start, &params)?;
        let mut out = prefix.to_vec();
        out.extend_from_slice(&suffix);
        Some(out)
    }

    // --- polish ----------------------------------------------------------------

    /// Local search on the action tape. Runs `chains` independent chains in
    /// parallel (different seeds) and returns the best result. Only exact,
    /// full-replay-validated tapes are accepted; `accept_equal` is the
    /// probability of accepting an equal-length neighbor (drift).
    pub fn polish(
        &self,
        tape: &[u8],
        iters: u64,
        chains: usize,
        seed: u64,
        accept_equal: f64,
    ) -> (Vec<u8>, u32) {
        let results: Vec<(Vec<u8>, u32)> = (0..chains.max(1))
            .into_par_iter()
            .map(|c| self.polish_chain(tape, iters, seed.wrapping_add(c as u64 * 7919), accept_equal))
            .collect();
        results
            .into_iter()
            .min_by_key(|&(_, t)| t)
            .unwrap()
    }

    fn polish_chain(&self, tape: &[u8], iters: u64, seed: u64, accept_equal: f64) -> (Vec<u8>, u32) {
        const CKPT_EVERY: usize = 32;

        let (completed, _, ticks) = self.replay(tape);
        if !completed {
            return (tape.to_vec(), u32::MAX);
        }
        let mut cur: Vec<u8> = tape[..ticks as usize].to_vec();
        let mut cur_ticks = ticks;
        let mut best = cur.clone();
        let mut best_ticks = cur_ticks;

        // Prefix checkpoints for fast partial replays.
        let mut ckpts: Vec<SimState> = build_checkpoints(self, &cur, CKPT_EVERY);
        let mut rng = Rng::new(seed);
        let mut cand: Vec<u8> = Vec::with_capacity(cur.len() + 8);

        for _ in 0..iters {
            let len = cur.len();
            if len < 4 {
                break;
            }
            let pos = rng.below(len as u64) as usize;
            cand.clear();
            cand.extend_from_slice(&cur);

            match rng.below(4) {
                0 => {
                    // delete 1..=3 ticks
                    let k = 1 + rng.below(3) as usize;
                    let k = k.min(cand.len() - pos);
                    cand.drain(pos..pos + k);
                }
                1 => {
                    // overwrite 1..=6 ticks with one action
                    let a = rng.below(6) as u8;
                    let k = (1 + rng.below(6) as usize).min(cand.len() - pos);
                    for t in cand[pos..pos + k].iter_mut() {
                        *t = a;
                    }
                }
                2 => {
                    // boundary shift: extend the run ending at pos over the next run
                    if pos + 1 < cand.len() {
                        let a = cand[pos];
                        let k = (1 + rng.below(4) as usize).min(cand.len() - pos - 1);
                        for t in cand[pos + 1..pos + 1 + k].iter_mut() {
                            *t = a;
                        }
                    }
                }
                _ => {
                    // insert 1..=2 ticks of a random action (enables rerouting;
                    // only survives if later deletions win the ticks back)
                    if cand.len() as u32 <= cur_ticks + 4 {
                        let a = rng.below(6) as u8;
                        let k = 1 + rng.below(2) as usize;
                        for _ in 0..k {
                            cand.insert(pos, a);
                        }
                    }
                }
            }

            // Replay from the closest checkpoint at or before the edit.
            let ck = (pos / CKPT_EVERY).min(ckpts.len().saturating_sub(1));
            let mut s = ckpts[ck];
            let start_tick = ck * CKPT_EVERY;
            let mut outcome_ticks: Option<u32> = None;
            for (i, &a) in cand[start_tick..].iter().enumerate() {
                match self.step(&mut s, a) {
                    StepOutcome::Completed => {
                        outcome_ticks = Some((start_tick + i + 1) as u32);
                        break;
                    }
                    StepOutcome::Crashed => break,
                    StepOutcome::Alive => {}
                }
            }

            if let Some(t) = outcome_ticks {
                let accept = t < cur_ticks || (t == cur_ticks && rng.unit_f64() < accept_equal);
                if accept {
                    cand.truncate(t as usize);
                    std::mem::swap(&mut cur, &mut cand);
                    cur_ticks = t;
                    ckpts = build_checkpoints(self, &cur, CKPT_EVERY);
                    if t < best_ticks {
                        best = cur.clone();
                        best_ticks = t;
                    }
                }
            }
        }
        (best, best_ticks)
    }
}

fn build_checkpoints(solver: &AceSolver, tape: &[u8], every: usize) -> Vec<SimState> {
    let mut out = Vec::with_capacity(tape.len() / every + 1);
    let mut s = solver.spawn;
    out.push(s);
    for (i, &a) in tape.iter().enumerate() {
        if solver.step(&mut s, a) != StepOutcome::Alive {
            break;
        }
        if (i + 1) % every == 0 {
            out.push(s);
        }
    }
    out
}

// --- geometry / grid helpers -------------------------------------------------

/// Exact segment intersection math from real_collision.rs::lines_intersect.
#[inline]
fn segments_intersect(
    p1x: f32, p1y: f32, p2x: f32, p2y: f32,
    q1x: f32, q1y: f32, q2x: f32, q2y: f32,
) -> bool {
    let s1x = p2x - p1x;
    let s1y = p2y - p1y;
    let s2x = q2x - q1x;
    let s2y = q2y - q1y;

    let denom = -s2x * s1y + s1x * s2y;
    if denom.abs() < 0.000001 {
        return false;
    }
    let s = (-s1y * (p1x - q1x) + s1x * (p1y - q1y)) / denom;
    let t = (s2x * (p1y - q1y) - s2y * (p1x - q1x)) / denom;
    s >= 0.0 && s <= 1.0 && t >= 0.0 && t <= 1.0
}

fn cell_of(rows: usize, cols: usize, hmin: (f32, f32), hcell: f32, x: f32, y: f32) -> (usize, usize) {
    let c = (((x - hmin.0) / hcell) as i64).clamp(0, cols as i64 - 1) as usize;
    let r = (((y - hmin.1) / hcell) as i64).clamp(0, rows as i64 - 1) as usize;
    (r, c)
}

/// Read a distance field with a small spiral fallback for blocked/unreached
/// cells (the ship's center can sit inside a wall's inflation zone).
fn spiral_read(field: &[i32], rows: usize, cols: usize, cell: (usize, usize), max_r: i32) -> i32 {
    let (r0, c0) = (cell.0 as i32, cell.1 as i32);
    for rad in 0..=max_r {
        let mut best = UNREACHABLE;
        for dr in -rad..=rad {
            for dc in -rad..=rad {
                if dr.abs() != rad && dc.abs() != rad {
                    continue;
                }
                let r = r0 + dr;
                let c = c0 + dc;
                if r < 0 || c < 0 || r >= rows as i32 || c >= cols as i32 {
                    continue;
                }
                let v = field[r as usize * cols + c as usize];
                if v >= 0 && v < best {
                    best = v;
                }
            }
        }
        if best < UNREACHABLE {
            return best + rad * 10;
        }
    }
    UNREACHABLE
}

fn build_blocked(
    rows: usize,
    cols: usize,
    hmin: (f32, f32),
    hcell: f32,
    lines: &[[f32; 4]],
    inflation: f32,
) -> Vec<bool> {
    let mut blocked = vec![false; rows * cols];
    let inf_sq = inflation * inflation;
    for l in lines {
        let (x1, y1, x2, y2) = (l[0], l[1], l[2], l[3]);
        let r0 = ((((y1.min(y2) - inflation) - hmin.1) / hcell) as i64).clamp(0, rows as i64 - 1) as usize;
        let r1 = ((((y1.max(y2) + inflation) - hmin.1) / hcell) as i64).clamp(0, rows as i64 - 1) as usize;
        let c0 = ((((x1.min(x2) - inflation) - hmin.0) / hcell) as i64).clamp(0, cols as i64 - 1) as usize;
        let c1 = ((((x1.max(x2) + inflation) - hmin.0) / hcell) as i64).clamp(0, cols as i64 - 1) as usize;
        for r in r0..=r1 {
            for c in c0..=c1 {
                if blocked[r * cols + c] {
                    continue;
                }
                let cx = hmin.0 + (c as f32 + 0.5) * hcell;
                let cy = hmin.1 + (r as f32 + 0.5) * hcell;
                if point_seg_dist_sq(cx, cy, x1, y1, x2, y2) < inf_sq {
                    blocked[r * cols + c] = true;
                }
            }
        }
    }
    blocked
}

fn point_seg_dist_sq(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    let t = if len_sq > 0.0 {
        (((px - x1) * dx + (py - y1) * dy) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let cx = x1 + t * dx;
    let cy = y1 + t * dy;
    let ex = px - cx;
    let ey = py - cy;
    ex * ex + ey * ey
}

fn dijkstra(
    rows: usize,
    cols: usize,
    hmin: (f32, f32),
    hcell: f32,
    blocked: &[bool],
    start_x: f32,
    start_y: f32,
) -> Vec<i32> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    const NEIGHBORS: [(i32, i32, i32); 8] = [
        (-1, 0, 10), (1, 0, 10), (0, -1, 10), (0, 1, 10),
        (-1, -1, 14), (-1, 1, 14), (1, -1, 14), (1, 1, 14),
    ];

    let mut dist = vec![-1i32; rows * cols];
    let (sr, sc) = cell_of(rows, cols, hmin, hcell, start_x, start_y);
    let mut heap: BinaryHeap<Reverse<(i32, usize, usize)>> = BinaryHeap::new();

    // Seed from the nearest unblocked cells if the start cell is blocked.
    if blocked[sr * cols + sc] {
        'outer: for rad in 1..=10i32 {
            let mut found = false;
            for dr in -rad..=rad {
                for dc in -rad..=rad {
                    if dr.abs() != rad && dc.abs() != rad {
                        continue;
                    }
                    let r = sr as i32 + dr;
                    let c = sc as i32 + dc;
                    if r < 0 || c < 0 || r >= rows as i32 || c >= cols as i32 {
                        continue;
                    }
                    let idx = r as usize * cols + c as usize;
                    if !blocked[idx] {
                        dist[idx] = rad * 10;
                        heap.push(Reverse((rad * 10, r as usize, c as usize)));
                        found = true;
                    }
                }
            }
            if found {
                break 'outer;
            }
        }
    } else {
        dist[sr * cols + sc] = 0;
        heap.push(Reverse((0, sr, sc)));
    }

    while let Some(Reverse((d, r, c))) = heap.pop() {
        if d > dist[r * cols + c] {
            continue;
        }
        for &(dr, dc, cost) in &NEIGHBORS {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= cols as i32 {
                continue;
            }
            let idx = nr as usize * cols + nc as usize;
            if blocked[idx] {
                continue;
            }
            let nd = d + cost;
            if dist[idx] == -1 || nd < dist[idx] {
                dist[idx] = nd;
                heap.push(Reverse((nd, nr as usize, nc as usize)));
            }
        }
    }
    dist
}
