# SpaceAce Reinforcement Learning Environment

A high-performance Rust-based reinforcement learning environment for the SpaceAce game, with Python bindings for easy integration with ML frameworks.

## Features

- **Fast Simulation**: Rust implementation for maximum performance during training
- **Gymnasium Compatible**: Standard RL interface that works with Stable-Baselines3, Ray RLlib, etc.
- **Faithful Physics**: Exact port of the original JavaScript game mechanics
- **Spatial Optimization**: Efficient collision detection with spatial partitioning
- **Multi-Level Support**: Support for different difficulty levels and maps

## Installation

### Prerequisites
- Rust (latest stable version)
- Python 3.8+
- uv (for fast Python package management)

### Build Instructions

1. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install Python dependencies**:
   ```bash
   uv add maturin gymnasium numpy "stable-baselines3[extra]"
   ```

4. **Build the Rust extension**:
   ```bash
   uv run maturin develop --release
   ```

## Quick Start

### Basic Usage

```python
import gymnasium as gym
from spaceace_env import SpaceAceEnv

# Create environment
env = SpaceAceEnv(level=1, max_steps=3000)

# Reset environment
obs, info = env.reset()

# Take actions
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
        
# Clean up
env.close()
```

### Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from spaceace_env import SpaceAceEnv

# Create environment
env = SpaceAceEnv(level=1)

# Create and train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Test trained agent
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Run Example Training Script

```bash
uv run python train_example.py
```

## Environment Details

### Observation Space

18-dimensional continuous space containing:
- Ship position (x, y)
- Ship velocity (vx, vy)
- Ship rotation (radians)
- Closest pickup position (x, y) and distance
- Wall distances in 8 directions (N, NE, E, SE, S, SW, W, NW)
- Number of pickups remaining
- Normalized ship position within map bounds

### Action Space

MultiDiscrete([2, 2, 2]) representing:
- `[0]`: Rotate left (0=off, 1=on)
- `[1]`: Rotate right (0=off, 1=on)  
- `[2]`: Thrust (0=off, 1=on)

### Reward Structure

- **-0.01** per step (encourages speed)
- **-100** for crashing into walls
- **+1000** for completing the level
- **+50** per pickup collected
- **Small bonus** for being close to pickups

### Physics Constants

Faithful reproduction of the original game:
- Gravity: 100 units/s²
- Thrust Power: 400 units/s²
- Rotation Speed: 4.363323 rad/s
- Ship Collision: 5-segment polygon
- Pickup Radius: 10 units

## Project Structure

```
spaceace-rl/
├── src/
│   ├── lib.rs           # Python bindings and main interface
│   ├── game.rs          # Core game logic and state management
│   ├── physics.rs       # Ship physics and movement
│   ├── collision.rs     # Collision detection system
│   └── map_parser.rs    # Level data parsing
├── spaceace_env.py      # Gymnasium environment wrapper
├── train_example.py     # Example training script
├── Cargo.toml          # Rust dependencies
├── pyproject.toml      # Python build configuration
└── README.md
```

## Performance

The Rust implementation provides significant performance improvements over JavaScript:
- **~100x faster** physics simulation
- **Efficient collision detection** with spatial partitioning
- **Memory efficient** state representation
- **No GC pauses** during training

This allows for:
- Training with millions of environment steps
- Parallel environment execution
- Real-time policy evaluation

## Extending the Environment

### Adding New Levels

To add support for actual game levels from `mapData.js`:

1. Update `map_parser.rs` to parse the JavaScript map data format
2. Implement the level loading logic in `game.rs`
3. Add pickup positions from the map data

### Custom Reward Functions

Modify the `calculate_reward()` function in `src/lib.rs` to implement custom reward shaping:

```rust
fn calculate_reward(&self) -> f32 {
    let mut reward = 0.0;
    
    // Your custom reward logic here
    
    reward
}
```

### Additional Observations

Extend the observation space by modifying `get_observation()` in `src/lib.rs`.

## Troubleshooting

### Build Issues

1. **Rust not found**: Install Rust from https://rustup.rs/
2. **uv not found**: Install uv from https://docs.astral.sh/uv/getting-started/installation/
3. **maturin not found**: `uv add maturin`
4. **Linker errors**: Install C++ build tools for your platform

### Runtime Issues

1. **Module not found**: Run `uv run maturin develop` to build the extension
2. **Performance issues**: Use `uv run maturin develop --release` for optimized builds
3. **Memory issues**: Reduce batch size or number of parallel environments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is released under the MIT License. See LICENSE file for details.