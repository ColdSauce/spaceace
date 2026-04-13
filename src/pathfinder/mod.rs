//! Pathfinding backends.
//!
//! `grid` is the canonical BFS-over-inflated-walls pathfinder used for reward
//! shaping, HRL planning, and MCTS basic mode. `momentum` wraps it with a
//! momentum-aware state-space search used by MCTS when finer motion planning
//! pays off. Callers pick one via `PathfinderKind`.
//!
//! Public API lives in the submodules; this mod.rs just re-exports the types
//! so the rest of the crate can still write `crate::pathfinder::PathfinderGrid`
//! and `crate::pathfinder::PathfinderKind` unchanged.

pub mod grid;
pub mod momentum;

pub use grid::PathfinderGrid;
pub use momentum::{MomentumPathfinder, PathfinderKind};
