# SpaceAce

Rust game engine + offline "Ace" planner that plays SpaceAce at superhuman
level by finding tick-perfect action tapes.

**Before working on the AI, read `docs/SOLVER.md`** — it explains the
design rationale, every failure mode already hit (don't re-hit them), the
diagnostic loop for improving the solver, and the open problems.

## Hard rules (user rulings)

- **No wall clipping.** Solve in `strict` mode (the default). The engine's
  every-other-frame collision skip above ~316px/s is exploitable but was
  ruled out for ghosts. `--allow-clip` is for experiments only.
- **No human-derived search guidance.** Never seed/corridor/bias the
  search with human ghost data — the user wants to learn from independent
  AI lines. Human ghosts are diagnostic benchmarks only.
- **Keep the game engine intact** (`src/real_*.rs`): all saved tapes and
  ghosts depend on exact float behavior. The solver's stepper must remain
  float-op-identical to the engine.

## Build & Run

```bash
# Build the Rust extension (required after any Rust code changes).
# The DYLD prefix works around Homebrew rustc being linked against LLVM 20
# while /opt/homebrew/opt/llvm points at LLVM 22 (rustc aborts otherwise).
# If cargo reports a stale dyld error afterwards: rm target/.rustc_info.json
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/llvm/20.1.8/lib
uv sync --reinstall-package spaceace-rl

# Solve a level (improves the stored tape/ghost when it finds a faster one)
uv run python scripts/solve.py --level 7 --budget-min 28          # standard
uv run python scripts/solve.py --level 7 --budget-min 60 --deep   # squeeze harder

# Watch the solved run / replay tapes
uv run python run.py --agent ace --level 7
uv run python run.py --agent tas --level 7 --tas-label tas --headless

# Race the ghosts in the browser (AI ghost = magenta)
uv run python -m dashboard    # then open http://localhost:5050/play

# Smoke tests (run to validate changes; they exercise level 7)
bash tests/smoke.sh
```

Do NOT use `maturin develop` — it conflicts with uv's environment management.
Always use `uv sync --reinstall-package spaceace-rl` to rebuild.

## Architecture

### Rust (`src/`)
- `lib.rs` — PyO3 bindings: `PyGameInstance` (engine), `PyPathfinder`
  (level tooling), `PySolver` (the AI)
- `real_game.rs` / `real_physics.rs` / `real_collision.rs` /
  `real_map_parser.rs` — the game. **Keep intact.**
- `solver.rs` — the entire AI in one file. Exact stepper, parallel beam
  search (route-DP heuristic, momentum-aware rank with fly-through credit
  and turnaround penalty, braking-feasibility doom model, mask-stratified
  selection), warm-started global/corridor and suffix refinement,
  exact-continuation window re-solves, local-search polish. Design details:
  `docs/SOLVER.md`.

### Python (`spaceace/`)
- `core/env.py` — `SpaceAceDirectEnv` wrapping `PyGameInstance`
- `core/viz.py` — pygame renderer
- `agents/` — `ace` (replays/plans solved tapes), `tas` (replays a
  sidecar), `random`, `human`
- `ghost_actions.py` — sidecar I/O (`ghost_actions/L{level}_tas.json`,
  ticks at 60/s, action indices 0-5)
- `scripts/solve.py` — the whole AI workflow in one command: portfolio
  solve → refine/exact-window/polish/suffix loop → validate on the real
  engine → save sidecar + dashboard ghosts

### Ghosts
- `dashboard/spaceace_dashboard.db` `ghost_replays`: best time per
  (level, ghost_type). `human` = the user's records; `tas`/`ai` = the
  AI's (the web UI renders `ai`). Overwritten only by strictly faster runs.
- A tape is only saved after an exact validation replay on `PyGameInstance`.

## Solver quick reference
- `scripts/solve.py --level N --budget-min M` is anytime: rerunning keeps
  improving the stored tape. `--fresh` ignores the incumbent sidecar.
- Debug telemetry: `ACE_DEBUG=1` prints per-beam-layer frontier stats.
- Key knobs: `width` (quality ∝ width), `lattice` (velocity-aware
  time-to-go rank — the strongest rank for ≤4-pickup levels; the driver
  uses it automatically), `mix`/`proj_div` (px-rank velocity reward
  strength/horizon), `quant_*` (dedup granularity), `doom_scale` (px-rank
  safety pressure; low inside warm refines, 2.0 for cautious fresh
  solves), `turn_w` (pickup-arrival alignment), `cell_strat_m`
  (selection-diversity protection per 128px cell × speed band).
- `scripts/analyze_tape.py --level N --out t.png` — per-leg splits, speed
  profile, slow sections, speed-colored trajectory render.
- Diagnostic loop for improving results: see `docs/SOLVER.md`.

## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` for the
full workflow. Use `bd` for ALL task tracking — do NOT use TodoWrite or
markdown TODO lists.

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim
bd close <id>
```

## Session Completion

**When ending a work session**, complete ALL steps below. Work is NOT
complete until `git push` succeeds.

1. File issues for remaining work
2. Run quality gates (`bash tests/smoke.sh`)
3. Update issue status
4. PUSH TO REMOTE: `git pull --rebase && bd dolt push && git push`
5. Verify `git status` shows "up to date with origin"
