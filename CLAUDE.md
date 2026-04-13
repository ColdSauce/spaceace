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

- `--agent {random,ppo,mcts}` — agent type
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
- `agents/ppo/` — PPO agent using stable-baselines3
- `agents/random_agent.py` — random baseline
- `env/direct.py` — `SpaceAceDirectEnv` wrapping Rust `PyGameInstance`
- `env/viz.py` — pygame renderer with HUD, minimap, debug path overlay

### Key Design Decisions
- MCTS uses heuristic leaf evaluation (pathfinder distance + velocity alignment + wall TTI), not random rollouts
- Pathfinder uses BFS on a 10px grid with 35px wall inflation for ship clearance
- Action space: 6 actions (coast, thrust, rotate_left, rotate_left+thrust, rotate_right, rotate_right+thrust)
- `action_repeat` groups frames into macro-actions for deeper lookahead per tree edge
- Game state save/load uses opaque Rust `GameSnapshot` objects (no serialization overhead)
