# Level 0 Training Guide - Ultra Simple SpaceAce

## Overview

**Level 0** is a custom-designed ultra-simple training level created specifically for RL agents to learn basic SpaceAce mechanics. It's dramatically easier than any existing level and should be used as the first step in curriculum learning.

## Level 0 Specifications

### Geometry
- **Vertices**: 6 total (minimal complexity)
- **Lines**: 4 total (just boundary walls)
- **Pickups**: 1 total (single objective)
- **Map Size**: 800×600 pixels (comfortable and manageable)

### Layout
```
+----------------------------------------+
|                                        |
|                                        |
| S                                   P  |
|                                        |
|                                        |
+----------------------------------------+
```
- **S**: Ship spawn at (200, 200) [safely inside boundary with 100+ pixel clearance]
- **P**: Single pickup at (550, 300) [clear path across]
- Simple rectangular boundary with no obstacles
- Boundary: (100,100) to (700,500) - plenty of room for learning
- 364-pixel distance to pickup (good for learning navigation)

### Complexity Comparison

| Level | Vertices | Lines | Pickups | Relative Difficulty |
|-------|----------|-------|---------|-------------------|
| **0** | **5** | **4** | **1** | **Ultra Easy 🟢** |
| 8 | 85 | 73 | 11 | Easy 🟡 |
| 6 | 139 | 131 | 7 | Easy 🟡 |
| 1 | 199 | 185 | 17 | Hard 🔴 |

Level 0 is **17× simpler** than Level 1 in terms of geometry complexity!

## Training Benefits

### Why Level 0 is Perfect for Initial Training

1. **Minimal Collision Risk**: Only 4 walls to avoid
2. **Single Clear Objective**: Just 1 pickup to collect
3. **Simple Navigation**: Straight-line path from spawn to pickup
4. **Fast Episodes**: Can complete in ~50-100 steps
5. **Clear Success Signal**: Easy to achieve +1000 completion reward

### Expected Learning Progression

**Phase 1 (Steps 0-10,000)**: Basic Control Learning
- Agent learns that actions affect movement
- Discovers thrust and rotation mechanics
- Begins to avoid immediate wall crashes

**Phase 2 (Steps 10,000-50,000)**: Survival Learning  
- Agent learns to stay alive for longer periods
- Develops basic collision avoidance
- Starts exploring the environment

**Phase 3 (Steps 50,000-100,000)**: Goal-Directed Behavior
- Agent discovers the pickup location
- Learns to navigate toward objectives
- Achieves first successful episode completions

**Phase 4 (Steps 100,000+)**: Optimization
- Agent optimizes path efficiency
- Minimizes episode length
- Achieves consistent success (>90% completion rate)

## Recommended Training Configuration

### Environment Setup
```python
from spaceace_env import SpaceAceEnv

# Use Level 0 for initial training
env = SpaceAceEnv(level=0, max_steps=500)  # Shorter episodes for faster learning
```

### Improved Reward Function
```python
def improved_reward_level_0(self, prev_pickup_dist, curr_pickup_dist):
    reward = 0.0
    
    # Survival bonus (positive reinforcement)
    reward += 0.5  # Higher survival bonus for simple level
    
    # Progress toward pickup (more generous)
    if curr_pickup_dist < prev_pickup_dist:
        reward += (prev_pickup_dist - curr_pickup_dist) * 0.5
    
    # Wall proximity penalty (gentler)
    min_wall_dist = min(self.get_wall_distances())
    if min_wall_dist < 30:  # Smaller danger zone
        reward -= (30 - min_wall_dist) * 0.01
    
    # Crash penalty (reduced)
    if crashed:
        reward -= 50.0  # Less harsh than -100
    
    # Pickup collection (keep high reward)
    reward += pickups_collected * 100.0
    
    # Completion bonus (keep high)
    if level_completed:
        reward += 1000.0
    
    return reward
```

### Algorithm Configuration
```python
# PPO Configuration optimized for Level 0
ppo_config = {
    "learning_rate": 1e-3,        # Higher LR for simple environment
    "n_steps": 1024,              # Shorter rollouts
    "batch_size": 32,             # Smaller batches
    "ent_coef": 0.2,              # Higher exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_epochs": 10,
    "policy_kwargs": {"net_arch": [128, 128]}  # Smaller network
}
```

## Expected Performance Metrics

### Success Criteria for Level 0
- **Episode Length**: <200 steps on average
- **Completion Rate**: >95% within 200,000 training steps
- **Average Reward**: >+900 per episode
- **Wall Collisions**: <5% of episodes

### Training Timeline
- **Initial Success**: Within 50,000-100,000 steps
- **Consistent Performance**: Within 200,000 steps
- **Optimal Performance**: Within 500,000 steps

### When to Graduate
Move to Level 8 when agent achieves:
- 95%+ completion rate over 1000 episodes
- Average episode length <150 steps
- Minimal wall collisions (<2%)

## Debugging and Monitoring

### Key Metrics to Track
```python
# Episode-level metrics
episode_length = []
completion_rate = []
average_reward = []
wall_collision_rate = []

# Step-level metrics  
survival_time = []
distance_to_pickup = []
wall_proximity = []
```

### Visualization Commands
```python
# ASCII render for debugging
print(env.render())

# Detailed state for analysis
print(env.render_detailed())
```

## Common Issues and Solutions

### Issue: Agent Never Finds Pickup
**Solution**: Increase exploration (higher entropy coefficient, random actions)

### Issue: Agent Crashes Into Walls Immediately  
**Solution**: Reduce physics difficulty, increase survival bonus

### Issue: Agent Learns to Hover but Never Moves
**Solution**: Add movement bonus, reduce survival bonus

### Issue: Training is Too Slow
**Solution**: Use smaller network, higher learning rate, shorter episodes

## Curriculum Progression

### Recommended Training Sequence
1. **Level 0**: Master basic controls and single pickup (Current)
2. **Level 8**: Simple geometry with multiple pickups  
3. **Level 6**: Smallest real level
4. **Level 7**: Fewest pickups (3)
5. **Level 5**: Medium complexity
6. **Level 4**: Medium-high complexity
7. **Level 2, 9, 1, 10, 3**: Advanced levels

### Transfer Learning
Once Level 0 is mastered, the learned control skills should transfer well to more complex levels. The agent will have learned:
- Basic thrust and rotation control
- Wall collision avoidance
- Goal-directed navigation
- Pickup collection mechanics

## Implementation Files

### Modified Files
- `spaceace_levels.json`: Added Level 0 data
- `spaceace_env.py`: Added Level 0 registration
- `test_level_0.py`: Comprehensive test script
- `verify_level_0.py`: Data verification script

### Usage Example
```python
import numpy as np
from spaceace_env import SpaceAceEnv

# Create Level 0 environment
env = SpaceAceEnv(level=0, max_steps=500)

# Train your agent
obs, info = env.reset()
for step in range(1000):
    # Your RL algorithm here
    action = your_agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
```

## Success Story Template

Your training logs should show progression like this:
```
Steps 0-25,000:     Episode length ~300, completion rate 0%
Steps 25,000-75,000: Episode length ~250, completion rate 10%  
Steps 75,000-150,000: Episode length ~200, completion rate 50%
Steps 150,000+:      Episode length ~150, completion rate 95%
```

**Level 0 is your foundation for SpaceAce mastery!** 🚀