# 🚀 SpaceAce RL Training Guide

Train RL agents on any SpaceAce level with flexible options!

## 🎯 Quick Start

### Basic Training (Level 1, 50k steps)
```bash
uv run python train_example.py
```

### Train on Different Levels
```bash
# Train on Level 4 (the one we fixed with proper spawn points)
uv run python train_example.py --level 4

# Train on challenging Level 10
uv run python train_example.py --level 10

# Any level 1-10
uv run python train_example.py --level 7
```

## ⚙️ Advanced Options

### Custom Training Duration
```bash
# Quick training (5k steps)
uv run python train_example.py --level 3 --timesteps 5000

# Long training (200k steps)
uv run python train_example.py --level 1 --timesteps 200000
```

### Episode Length Control
```bash
# Shorter episodes for difficult levels
uv run python train_example.py --level 8 --max-steps 1000

# Longer episodes for exploration
uv run python train_example.py --level 1 --max-steps 5000
```

### Skip Components for Faster Training
```bash
# Skip random baseline test
uv run python train_example.py --level 5 --skip-random

# Skip final testing (train only)
uv run python train_example.py --level 2 --skip-test

# Skip both (fastest training)
uv run python train_example.py --level 6 --skip-random --skip-test
```

## 📊 Monitor Progress

### View Training Graphs
```bash
uv run tensorboard --logdir ./tensorboard_logs/
# Open http://localhost:6006 in browser
```

### Watch Trained Agent
```bash
# Watch the agent play (after training)
uv run python watch_agent.py --mode trained --level 4

# Compare with random agent
uv run python watch_agent.py --mode compare --level 4
```

## 📁 Output Files

Training creates organized outputs:
```
./models/level_X/
├── best_model.zip          # Best performing model during training
└── ppo_spaceace_final.zip  # Final trained model

./logs/level_X/
└── evaluations.npz         # Training evaluation data

./tensorboard_logs/
└── PPO_X/                  # Tensorboard training logs
```

## 🎮 Level Characteristics

- **Level 1**: Good starting level, 185 map lines, 17 pickups
- **Level 2**: Similar complexity, 179 map lines  
- **Level 3**: Complex combined level, 266 map lines
- **Level 4**: Fixed spawn point, 130 map lines, 13 pickups
- **Level 5-10**: Various difficulties and geometries

## 🏆 Expected Performance

- **Random Agent**: ~-65 reward (crashes quickly)
- **Trained Agent**: 100+ reward (successful navigation)
- **Training Speed**: 6,000+ steps/second
- **Training Time**: 5-15 minutes for 50k steps

## 💡 Tips

1. **Start with Level 1** to verify training works
2. **Use Level 4** to test fixed spawn points
3. **Try shorter episodes** (--max-steps 1000) for difficult levels
4. **Monitor tensorboard** for training progress
5. **Use --skip-random --skip-test** for fastest training

## 🔧 All Flags

```bash
uv run python train_example.py \
  --level 4 \              # Level to train on (1-10)
  --timesteps 50000 \      # Training steps
  --max-steps 3000 \       # Max steps per episode  
  --skip-random \          # Skip random baseline
  --skip-test              # Skip final testing
```