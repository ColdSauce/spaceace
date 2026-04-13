# Plan 4: Wire `backend` arg through `PyPathfinder`

## Context

Phase 6 of the re-architecture moved `pathfinder.rs` and `momentum_pathfinder.rs` under `src/pathfinder/{grid,momentum}.rs` but did not unify the PyO3 surface. Today:

- `PyPathfinder::new(level)` in `src/lib.rs` always constructs a `PathfinderGrid`. There is no momentum variant exposed to Python.
- `spaceace/strategies/pathfinder.py::RustPathfinder.__init__(level, backend="grid")` accepts a `backend` argument and then **ignores it** — comment at the top of that file calls it out.
- MCTS dispatches between grid and momentum internally through `PyMCTSEngine(level, max_steps, use_momentum)`; this is a completely separate code path.

The re-architecture plan approved "unify behind one PyO3 class, keep both backends." This plan finishes that work.

## Key question: is this actually needed?

Quick audit of callers of `RustPathfinder` and `spaceace_rl.PyPathfinder`:

- `spaceace/strategies/pathfinder.py` — only calls `get_nearest_pickup_info`
- `spaceace/strategies/observation.py::PathAugmentedObs23` — uses the strategy above
- `spaceace/strategies/rewards.py::DenseShapedReward` — uses the strategy above
- `spaceace/agents/hrl/agent.py` — calls `get_pickup_coords`, `held_karp_tsp`, `get_path_to_specific_pickup`, `get_distance_to_specific_pickup`
- `spaceace/agents/hrl/waypoint_env.py` — calls `get_pickup_coords`, `get_distance_to_specific_pickup`

**Nobody currently constructs a momentum pathfinder from Python.** The grid path covers every real use case. This plan's payoff is mostly architectural hygiene: remove the lying comment in `RustPathfinder`, make the backend selector honest, and leave the door open for future callers. Effort should be small; if it starts ballooning, stop and just delete the unused `backend` kwarg instead.

## Audit `MomentumPathfinder`'s public API

Open `src/pathfinder/momentum.rs` and enumerate which methods exist:

```
grid methods used from Python:
  get_nearest_pickup_info(ship_x, ship_y, collected) -> (dist, dir_x, dir_y)
  get_distance_to_specific_pickup(ship_x, ship_y, pickup_idx) -> (dist, dir_x, dir_y)
  held_karp_tsp(ship_x, ship_y, collected) -> Vec<usize>
  get_path_to_specific_pickup(ship_x, ship_y, pickup_idx) -> Vec<(f32, f32)>
  get_pickup_coords() -> Vec<(f32, f32)>
  get_debug_target_info(ship_x, ship_y, collected) -> (...)
```

For each, check: does `MomentumPathfinder` already implement it? (Likely: `get_nearest_pickup_info`, `get_debug_target_info`. Likely not: `held_karp_tsp`, `get_path_to_specific_pickup`, `get_pickup_coords`.)

The audit decides the design.

## Design

Two layers.

### Rust side

Change the PyO3 class from owning a concrete `PathfinderGrid` to owning the existing `PathfinderKind` enum:

```rust
// src/lib.rs
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
        game.load_level(level).map_err(...)?;
        let kind = match backend {
            "grid" => PathfinderKind::Grid(PathfinderGrid::build(&game)),
            "momentum" => PathfinderKind::Momentum(MomentumPathfinder::new(&game)),
            other => return Err(PyValueError::new_err(format!("unknown backend: {other}"))),
        };
        Ok(PyPathfinder { kind })
    }

    fn backend(&self) -> &'static str {
        match self.kind {
            PathfinderKind::Grid(_) => "grid",
            PathfinderKind::Momentum(_) => "momentum",
        }
    }

    fn get_nearest_pickup_info(&self, x: f32, y: f32, collected: Vec<bool>) -> (f64, f64, f64) {
        self.kind.get_nearest_pickup_info(x, y, &collected)  // PathfinderKind already does this dispatch
    }

    // Methods that ONLY grid supports:
    fn held_karp_tsp(&self, x: f32, y: f32, collected: Vec<bool>) -> PyResult<Vec<usize>> {
        match &self.kind {
            PathfinderKind::Grid(g) => Ok(g.held_karp_tsp(x, y, &collected)),
            PathfinderKind::Momentum(_) => Err(PyNotImplementedError::new_err(
                "held_karp_tsp is only available on the grid backend",
            )),
        }
    }
    // ... same for get_path_to_specific_pickup, get_pickup_coords, etc.
}
```

Guard clauses beat a shrunken common trait: they preserve HRL's current behavior exactly, and momentum callers (when they appear) get a clear error telling them what's missing.

### Python side

```python
# spaceace/strategies/pathfinder.py
class RustPathfinder:
    def __init__(self, level: int, backend: str = "grid"):
        if backend not in ("grid", "momentum"):
            raise ValueError(f"unknown backend {backend!r}")
        self.level = level
        self.backend = backend
        self._impl = spaceace_rl.PyPathfinder(level, backend)

    def nearest_pickup_info(self, x, y, collected):
        return self._impl.get_nearest_pickup_info(x, y, collected)
```

Delete the "Phase 6 will wire this through" comment. Add a registry entry so callers can construct via config:

```python
# spaceace/strategies/__init__.py
STRATEGY_REGISTRY["pathfinder"] = {
    "rust_grid": lambda level: RustPathfinder(level, backend="grid"),
    "rust_momentum": lambda level: RustPathfinder(level, backend="momentum"),
}
```

## Files

**Edit**
- `src/lib.rs` — replace `PyPathfinder` struct + impl
- `spaceace/strategies/pathfinder.py` — honor `backend`
- `spaceace/strategies/__init__.py` — add momentum registry entry
- `tests/smoke.sh` — add a one-liner that constructs `RustPathfinder(0, backend="momentum")` and calls `nearest_pickup_info` to catch regressions

**No changes** to:
- `src/mcts.rs`, `src/alphazero_mcts.rs` — they use `PathfinderKind` directly, not through `PyPathfinder`
- HRL agent code — still constructs with default `"grid"` backend, still works

## Risks

- **`MomentumPathfinder::new` constructor shape**: may take different args than `PathfinderGrid::build`. Check before assuming the `match` arm compiles.
- **Method availability surprise**: if Momentum secretly *does* support more than expected, tighten the guards later rather than expand them now.
- **Behavioral regression**: since no Python caller uses momentum today, any bug in the new path goes unnoticed. The smoke line (`RustPathfinder(0, backend="momentum").nearest_pickup_info(...)`) covers construction; if an agent ever starts using momentum for real, add a targeted test.

## Verification

```bash
# 1. Rust rebuild.
uv sync --reinstall-package spaceace-rl

# 2. Construction smoke (both backends).
uv run python -c "
from spaceace.strategies import RustPathfinder
grid = RustPathfinder(0, backend='grid')
mom = RustPathfinder(0, backend='momentum')
print(grid.nearest_pickup_info(500.0, 500.0, [False]*10))
print(mom.nearest_pickup_info(500.0, 500.0, [False]*10))
"

# 3. Guard check: grid-only method raises on momentum backend.
uv run python -c "
import spaceace_rl
pf = spaceace_rl.PyPathfinder(0, 'momentum')
try:
    pf.held_karp_tsp(500.0, 500.0, [False]*10)
except NotImplementedError as e:
    print('guard OK:', e)
"

# 4. Full smoke — HRL path should be unaffected.
tests/smoke.sh
```

## Effort

**Small.** ~half a day. Mostly Rust plumbing; one rebuild cycle.

**Prerequisite:** audit `src/pathfinder/momentum.rs` first to confirm constructor signature and method availability. If Momentum lacks `get_nearest_pickup_info` too, this plan is wasted effort and we should instead delete the `backend` kwarg in `RustPathfinder` altogether.
