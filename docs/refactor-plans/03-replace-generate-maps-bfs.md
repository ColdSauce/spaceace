# Plan 3: Replace inline BFS in `generate_maps.py` with `RustPathfinder`

## Context

`spaceace/tools/generate_maps.py` (~1100 lines) procedurally generates SpaceAce levels and validates that all pickups are reachable from spawn. The validator re-implements the same BFS/grid-inflation/point-to-segment-distance math that lives in `src/pathfinder/grid.rs`, so there are two copies of the reachability algorithm that have to stay in sync — they do *not*, which is exactly the source of bugs that have shipped before.

The 1100 lines split roughly as:
- **~150 lines of reachability math** (the target of this plan): `point_to_segment_dist_sq`, `MapValidator._build_blocked_grid`, `MapValidator._bfs`, `MapValidator.validate`, `MapValidator.compute_distances`.
- **~950 lines of generation heuristics**: `RoomCorridorGenerator`, `MazeGenerator`, corridor-width measurement, difficulty scoring, CLI. These stay untouched.

The catch: generation is iterative — generate, validate, reject, regenerate. Validation has to be fast (tens to hundreds of calls per generated level). A Python-boundary PyO3 call is cheap (<1ms), so this is fine.

## Design

### New Rust methods on `PyPathfinder`

Two methods need to exist on the grid backend:

```rust
// src/pathfinder/grid.rs -> exposed via src/lib.rs PyPathfinder
fn validate_reachability(&self, spawn_x: f32, spawn_y: f32) -> (bool, Vec<f64>)
// Returns (all_reachable, per_pickup_path_distances). Unreachable pickups get f64::INFINITY.

fn distance_grid(&self) -> Vec<Vec<f64>>  // optional, for difficulty scoring
// Returns the raw BFS distance grid (cells × CELL_SIZE) so Python's difficulty
// scorer can compute the same metrics it does today without running its own BFS.
```

`PathfinderGrid::build(&game)` already performs BFS during construction. These methods just expose the results.

### New Rust constructor for freshly-generated levels

Currently `PyPathfinder::new(level)` loads a level by index from `data/`. Generator needs to pathfind over a level that is not yet on disk. Add a second constructor:

```rust
#[staticmethod]
fn from_map_json(map_json: &str) -> PyResult<PyPathfinder>
```

It parses the same JSON format `real_map_parser.rs` already handles, builds a `RealSpaceAceGame` in-memory, and constructs the grid. No temp files, no disk writes.

### Python side

```python
# spaceace/tools/generate_maps.py

import spaceace_rl

class MapValidator:
    CELL_SIZE = 10.0
    INFLATION_RADIUS = 35.0

    def __init__(self, level_json: str):
        # Old _build_blocked_grid / _bfs deleted.
        self._pf = spaceace_rl.PyPathfinder.from_map_json(level_json)

    def validate(self, spawn_x, spawn_y) -> tuple[bool, list[float]]:
        return self._pf.validate_reachability(spawn_x, spawn_y)

    def compute_distances(self, spawn_x, spawn_y) -> list[float]:
        _, dists = self._pf.validate_reachability(spawn_x, spawn_y)
        return dists
```

`point_to_segment_dist_sq` and `compute_min_corridor_width` stay in Python — they're generation-time geometry, not pathfinding. Keep them. Only the BFS math goes away.

## Constants audit

`CELL_SIZE = 10.0` and `INFLATION_RADIUS = 35.0` must match between Python and Rust. Before deleting the Python BFS:

1. Grep Rust for both constants (`src/pathfinder/grid.rs`). Confirm they are 10.0 and 35.0 exactly.
2. Check whether Rust uses cell-center or cell-corner sampling and whether it counts diagonal neighbors at √2 cost or 1 cost. Whatever it does, the Python version must have been doing — if there's divergence, a previously-validated level may fail validation post-port.

If the algorithms don't match, the correct answer is to *fix Python to match Rust* (Rust is the canonical one used by MCTS/reward shaping). That's a different problem; flag it during the audit.

## Migration

1. Audit constants + algorithm (30 min of reading).
2. Add `validate_reachability` and `from_map_json` to `src/pathfinder/grid.rs` + `src/lib.rs` PyO3 surface. Rebuild.
3. Add unit test: construct `PyPathfinder` two ways — `::new(0)` and `::from_map_json(data/levels/level0.json)` — assert `validate_reachability` returns identical distances.
4. In `generate_maps.py`, keep the old `MapValidator` class but route through Rust. Run the full generation suite end-to-end (generate 50 levels of each strategy, compare validation outcomes vs. baseline).
5. If outputs match, delete the old `_build_blocked_grid` / `_bfs` / internal constants.
6. Optionally expose `distance_grid()` and swap the difficulty scorer to use it.

## Files

**Edit**
- `src/pathfinder/grid.rs` — add `validate_reachability`, `from_map_json_data` helpers
- `src/lib.rs` — expose methods on `PyPathfinder`
- `spaceace/tools/generate_maps.py` — delete ~150 lines of BFS, route through `PyPathfinder`

**Create**
- `tests/test_pathfinder_parity.py` — asserts new Rust methods match the old Python ones on a fixed set of levels

## Risks

- **Algorithmic drift**: the #1 risk. If Python's BFS diagonally moves at cost 1 while Rust moves at cost √2, reachability on narrow corridors will disagree. Fix by aligning to Rust, not by keeping the divergence.
- **Silent map rejection regression**: generated maps that used to pass now fail (or vice versa). Run the generator in both modes against a seed of 100 levels and diff the accept/reject decisions. If they differ, investigate before merging.
- **JSON parse cost**: `from_map_json` on hot loops adds overhead. Measure; if bad, add a cache keyed on `hash(json)`.
- **`corridor width` calculation**: unrelated to BFS but shares `point_to_segment_dist_sq`. Leave it alone.

## Verification

```bash
# 1. Rust parity test on known levels.
uv run pytest tests/test_pathfinder_parity.py

# 2. Generate a seed of 50 maps with each strategy, both before and after.
uv run python -m spaceace.tools.generate_maps --strategy room --count 50 --seed 42 --output /tmp/maps_after.json
# Compare /tmp/maps_after.json to a pre-port baseline saved in the same repo.
diff <(jq -S . docs/refactor-plans/baselines/generate_maps_seed42.json) <(jq -S . /tmp/maps_after.json)

# 3. Full smoke (ensures MCTS/reward shaping still work on real levels).
tests/smoke.sh
```

## Effort

**Medium.** 1–2 days. The Rust side is ~50 lines of new code. The Python side is a ~150-line deletion. The real work is the parity audit — if algorithms diverge, expect an extra day of alignment.
