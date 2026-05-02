# Agent instructions

Use `bd` (beads) for all task tracking. Do not use Markdown plan files.

- `bd ready` — what's actionable
- `bd create "title" -t task -p 1` — file new work
- `bd show <id>` — read details
- `bd update <id> --status in_progress`
- `bd close <id> --reason "..."`
- `bd --json` — machine-readable output for parsing

# SpaceAce RL

Rust game engine + Python RL pipeline for the SpaceAce arcade game.

## Build & Run

```bash
# Build the Rust extension (required after any Rust code changes)
uv sync --reinstall-package spaceace-rl

# Run an agent with visualization
uv run python run.py --agent mcts --level 6 --num-simulations 5000

# Run headless (no pygame window)
uv run python run.py --agent mcts --level 6 --headless --episodes 10

# Train PPO agent
uv run python train.py --level 0
```

Do NOT use `maturin develop` — it conflicts with uv's environment management. Always use `uv sync --reinstall-package spaceace-rl` to rebuild.

## CLI Options (run.py)

- `--agent {random,ppo,mcts,alphazero}` — agent type
- `--level N` — game level (0-7)
- `--episodes N` — number of episodes (default 5)
- `--headless` — disable pygame visualization
- `--num-simulations N` — MCTS sims per decision (default 200, use 2000-5000 for good play)
- `--exploration F` — UCT exploration constant (default 1.41)
- `--fps N` — target FPS (default 60)
- `--max-steps N` — max steps per episode (default 3000)
- `--model PATH` — model path for PPO agent

## Architecture

### Rust Engine (`src/`)
- `lib.rs` — PyO3 bindings: `PyGameInstance`, `PyGameState`, `PyMCTSEngine`
- `real_game.rs` — game state, physics stepping, save/load via `GameSnapshot`
- `real_physics.rs` — ship physics (thrust, rotation, velocity)
- `real_collision.rs` — line-segment collision detection
- `real_map_parser.rs` — parses level JSON from `data/`
- `mcts.rs` — MCTS tree search with heuristic evaluation (no rollouts)
- `pathfinder.rs` — BFS grid pathfinder with wall inflation, greedy TSP for pickup ordering

### Python (`spaceace/`)
- `agents/base.py` — `BaseAgent` interface
- `agents/mcts/agent.py` — MCTS agent (delegates to Rust `PyMCTSEngine`)
- `agents/alphazero/` — AlphaZero PUCT MCTS + neural network evaluator
- `agents/ppo/` — PPO agent using stable-baselines3
- `agents/random_agent.py` — random baseline
- `env/direct.py` — `SpaceAceDirectEnv` wrapping Rust `PyGameInstance`
- `env/viz.py` — pygame renderer with HUD, minimap, debug path overlay

## Inspecting training jobs / runs

Training is logged in three places: `dashboard/job_logs/job_<id>.log` (raw
stdout), `tensorboard_logs/<run_name>/` (SB3 scalars), and the SQLite DB at
`dashboard/spaceace_dashboard.db` (jobs, runs, metric snapshots, checkpoints).

For debugging, prefer the unified read-only CLI rather than grepping logs by
hand — it pulls from all three sources and is friendly to LLM consumption:

```bash
uv run python scripts/inspect_run.py list                  # recent jobs + runs
uv run python scripts/inspect_run.py job  <job-id>         # full picture: args, errors, log tail, linked TB run
uv run python scripts/inspect_run.py run  <run-name|id>    # metric summary, linked job, checkpoints
uv run python scripts/inspect_run.py errors <job-id>       # tracebacks only
uv run python scripts/inspect_run.py tail   <job-id> -n 80
```

Pass `--json` on any subcommand for machine-readable output.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
