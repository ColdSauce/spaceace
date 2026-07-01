//! Grid pathfinder: BFS over inflated walls. Used by the level tools
//! (reachability validation, difficulty analysis) via `PyPathfinder`.
//! The Ace solver (src/solver.rs) builds its own distance fields.

pub mod grid;

pub use grid::{PathfinderGrid, DifficultyMetrics};
