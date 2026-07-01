# SpaceAce

Rust engine + offline planner ("Ace") for the SpaceAce arcade game. The AI
finds tick-perfect action tapes that complete each level faster than the best
human times, and the game/dashboard replays them as ghosts.

## Setup

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Build Rust extension + install Python deps
uv sync --reinstall-package spaceace-rl
```

**Important:** Do not use `maturin develop` — it conflicts with uv's
environment. Always rebuild with `uv sync --reinstall-package spaceace-rl`.

## Usage

```bash
# Solve a level (beam search + refinement; saves ghost + sidecar if faster)
uv run python scripts/solve.py --level 7 --budget-min 20

# Watch the solved run
uv run python run.py --agent ace --level 7

# Play with keyboard
uv run python play.py --level 7

# Smoke tests
bash tests/smoke.sh
```

## How the AI works

Everything lives in `src/solver.rs` (one file) and is orchestrated by
`scripts/solve.py`:

1. **Exact simulation** — a compact, allocation-free copy of the game step
   that matches the engine float-for-float, so a planned tape replays
   identically in the real game (validated on `PyGameInstance` before
   saving). ~6M steps/s per core.
2. **Global beam search** — tick-synchronized beam over the full state
   (position, velocity, rotation, pickup set), ranked by per-pickup Dijkstra
   distance fields plus an exact DP over remaining-pickup subsets (optimal
   route lower bound). A physical "can it still brake?" doom model prunes
   ballistically dead states; per-pickup-set stratification keeps alternate
   routes alive.
3. **Anytime refinement** — warm-started corridor re-search around the best
   tape at finer quantization (guaranteed never worse), exact local-search
   polish on the action tape, and suffix re-solves. Runs until the time
   budget expires.

The `ace` agent replays the solved tape (`ghost_actions/L{level}_tas.json`);
`scripts/solve.py` also stores dashboard ghosts (`tas` and `ai` labels) when
it beats the stored time.

## Layout

```
src/                 Rust engine (PyO3 bindings in lib.rs)
  real_game.rs       game loop, pickups, win/crash
  real_physics.rs    ship physics (thrust 400, gravity 100, rot 4.36 rad/s)
  real_collision.rs  segment collision w/ 500px spatial grid
  real_map_parser.rs level JSON parsing
  pathfinder/        BFS grid pathfinder (level tooling only)
  solver.rs          the AI: exact sim + beam search + refinement + polish

spaceace/
  core/              SpaceAceDirectEnv (Rust wrapper), gym wrapper, pygame viz
  agents/            ace (solver playback), tas (sidecar replay), random, human
  strategies/        canonical 6-action table
  ghost_actions.py   exact per-tick action sidecars (ghost_actions/*.json)
  tools/             level generation / analysis utilities

scripts/solve.py     the solver driver (portfolio → refine/polish loop → save)
dashboard/           Flask dashboard: ghosts, leaderboard, job history
web/                 browser build of the game with ghost racing
```

## License

MIT
