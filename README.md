# SpaceAce RL

Rust game engine + Python RL pipeline for the SpaceAce arcade game. Multiple agents (MCTS, PPO, AlphaZero, HRL) share composable strategies for pathfinding, observations, rewards, and actions via a plugin registry.

## Setup

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Build Rust extension + install Python deps
uv sync --reinstall-package spaceace-rl
```

**Important:** Do not use `maturin develop` — it conflicts with uv's environment. Always rebuild with `uv sync --reinstall-package spaceace-rl`.

## Usage

```bash
# Run MCTS agent with visualization
uv run python run.py --agent mcts --level 0 --num-simulations 5000

# Run headless
uv run python run.py --agent random --level 0 --headless --episodes 10

# Train PPO
uv run python train.py --level 0

# Play with keyboard
uv run python play.py --level 0
```

### CLI options

| Flag | Description |
|------|-------------|
| `--agent {random,human,mcts,ppo,alphazero,hrl}` | Agent type |
| `--level N` | Game level (0-7) |
| `--episodes N` | Number of episodes |
| `--headless` | Disable pygame window |
| `--num-simulations N` | MCTS/AlphaZero sims per decision |
| `--momentum-pathfinder` | Use momentum pathfinder backend |
| `--max-steps N` | Max steps per episode |
| `--model PATH` | Model checkpoint for PPO |

## Architecture

```
spaceace/
  core/           # SpaceAceDirectEnv (Rust wrapper), GymWrapper, pygame renderer
  strategies/     # Composable building blocks: Pathfinder, ObservationBuilder,
                  #   RewardShaper, ActionSpace + STRATEGY_REGISTRY
  agents/         # Agent implementations + AGENT_REGISTRY
    random_agent.py, human.py
    mcts/         # Rust MCTS with heuristic evaluation
    ppo/          # SB3 PPO inference + training env
    alphazero/    # Rust PUCT MCTS + neural net
    hrl/          # Hierarchical: TSP planner + pilot DQN
  training/       # Trainer ABC, Sb3Trainer, callbacks, vec-env factories
                  #   + TRAINER_REGISTRY

src/              # Rust engine (PyO3 bindings)
  lib.rs          # PyGameInstance, PyMCTSEngine, PyAlphaZeroEngine, PyPathfinder
  real_game.rs    # Game state + physics stepping
  real_physics.rs # Ship physics (thrust, rotation, gravity)
  real_collision.rs  # Line-segment collision detection
  real_map_parser.rs # Level JSON parsing
  mcts.rs         # MCTS tree search with heuristic leaf eval
  pathfinder/     # BFS grid + momentum pathfinder backends
```

### Key design decisions

- **Strategy pattern**: Agents consume reusable `Pathfinder`, `ObservationBuilder`, `RewardShaper`, and `ActionSpace` objects. New strategies register in `STRATEGY_REGISTRY` and are available to all agents/trainers.
- **Plugin registries**: `AGENT_REGISTRY`, `TRAINER_REGISTRY`, `STRATEGY_REGISTRY` with `@register_agent` / `@register_trainer` decorators.
- **Rust performance**: Game physics, MCTS, pathfinding, and collision detection run in Rust via PyO3. No serialization overhead (opaque `GameSnapshot` for state save/load).
- **Action space**: 6 discrete macro-actions over `[rot_left, rot_right, thrust]` with configurable `action_repeat`.

## Tests

```bash
# Smoke tests (all agents, ~60s)
bash tests/smoke.sh

# Unit tests
uv run pytest tests/ -v
```

## License

MIT
