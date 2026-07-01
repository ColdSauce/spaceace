# SpaceAce

Rust game engine + offline "Ace" planner that plays SpaceAce at superhuman
level by finding tick-perfect action tapes.

## Build & Run

```bash
# Build the Rust extension (required after any Rust code changes).
# The DYLD prefix works around Homebrew rustc being linked against LLVM 20
# while /opt/homebrew/opt/llvm points at LLVM 22 (rustc aborts otherwise).
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/llvm/20.1.8/lib
uv sync --reinstall-package spaceace-rl

# Solve a level (improves the stored tape/ghost when it finds a faster one)
uv run python scripts/solve.py --level 7 --budget-min 20

# Watch the solved run / replay tapes
uv run python run.py --agent ace --level 7
uv run python run.py --agent tas --level 7 --tas-label tas --headless

# Smoke tests (run these to validate changes; they exercise level 7)
bash tests/smoke.sh
```

Do NOT use `maturin develop` — it conflicts with uv's environment management.
Always use `uv sync --reinstall-package spaceace-rl` to rebuild.

## Architecture

### Rust (`src/`)
- `lib.rs` — PyO3 bindings: `PyGameInstance` (engine), `PyPathfinder`
  (level tooling), `PySolver` (the AI)
- `real_game.rs` / `real_physics.rs` / `real_collision.rs` /
  `real_map_parser.rs` — the game. **Keep intact**: ghosts and solved tapes
  depend on exact physics.
- `solver.rs` — the entire AI in one file: an exact allocation-free copy of
  the game step (must stay float-op-identical to real_physics/real_game),
  parallel beam search (per-pickup Dijkstra fields + exact subset-DP route
  lower bound, braking-feasibility doom model, mask-stratified selection),
  warm-started corridor refinement, suffix re-solve, and local-search polish.

### Python (`spaceace/`)
- `core/env.py` — `SpaceAceDirectEnv` wrapping `PyGameInstance`
- `core/viz.py` — pygame renderer
- `agents/` — `ace` (replays/plans solved tapes), `tas` (replays a sidecar),
  `random`, `human`
- `ghost_actions.py` — sidecar I/O (`ghost_actions/L{level}_tas.json`,
  ticks at 60/s, action indices 0-5)
- `scripts/solve.py` — solver driver: portfolio solve → refine/polish/suffix
  loop → validate on the real engine → save sidecar + dashboard ghosts

### Ghosts
- `dashboard/spaceace_dashboard.db` `ghost_replays` table: best time per
  (level, ghost_type). `human` rows are the user's records; `tas`/`ai` rows
  are the AI's. `scripts/solve.py` only overwrites when strictly faster.
- A tape is only saved after an exact validation replay on `PyGameInstance`.

## Solver notes
- Sim exactness is everything: any change to the Rust engine invalidates all
  saved tapes. `PySolver.replay()` must agree with `PyGameInstance` replay;
  the smoke test checks the L7 sidecar for exactly this.
- `scripts/solve.py --level N --budget-min M` is the whole workflow. It is
  anytime: rerunning with a bigger budget keeps improving the stored tape.
- Beam params that matter: `width` (quality ∝ width), `mix`/`proj_div`
  (velocity reward strength/horizon), `quant_*` (dedup granularity).

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
