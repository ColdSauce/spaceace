# SpaceAce Observation Space Documentation

The SpaceAce environment uses a 19-dimensional continuous observation vector. Here's the exact breakdown of what each dimension contains:

## Observation Vector Structure (19 dimensions)

### Ship State (5 values)
- **[0]** `ship_x` - Ship's X position in world coordinates
- **[1]** `ship_y` - Ship's Y position in world coordinates  
- **[2]** `ship_vx` - Ship's X velocity
- **[3]** `ship_vy` - Ship's Y velocity
- **[4]** `ship_rotation` - Ship's rotation angle in radians

### Closest Pickup Information (3 values)
- **[5]** `closest_pickup_x` - X position of the nearest uncollected pickup
- **[6]** `closest_pickup_y` - Y position of the nearest uncollected pickup
- **[7]** `closest_pickup_dist` - Euclidean distance to the nearest uncollected pickup

### Wall Distances (8 values)
Eight raycast distances to the nearest walls in the following directions:
- **[8]** North (0°, -1)
- **[9]** Northeast (45°, 0.707, -0.707)
- **[10]** East (90°, 1, 0)
- **[11]** Southeast (135°, 0.707, 0.707)
- **[12]** South (180°, 0, 1)
- **[13]** Southwest (225°, -0.707, 0.707)
- **[14]** West (270°, -1, 0)
- **[15]** Northwest (315°, -0.707, -0.707)

Each wall distance is calculated using a raycast from the ship's position with a maximum distance of 1000.0 units.

### Game State (1 value)
- **[16]** `pickups_remaining` - Number of uncollected pickups (as float)

### Normalized Position (2 values)
- **[17]** `normalized_x` - Ship's X position normalized to [0, 1] within map bounds
- **[18]** `normalized_y` - Ship's Y position normalized to [0, 1] within map bounds

## Implementation Details

The observation is constructed in the Rust code at `/Users/coldsauce/projects/spaceace/src/lib.rs` in the `get_observation` method (lines 83-117).

### Key Methods Used:
1. `game.get_state()` - Returns ship position, velocity, rotation, and game status
2. `game.get_closest_pickup()` - Finds the nearest uncollected pickup and its distance
3. `game.get_wall_distances()` - Performs 8 raycasts to find wall distances
4. `game.get_map_bounds()` - Gets map boundaries for position normalization

### Special Cases:
- If no pickups remain, closest pickup position defaults to ship's current position with distance 0
- Wall distances default to 1000.0 if no wall is found within that range
- Normalized positions are calculated as: `(position - min_bound) / (max_bound - min_bound)`

## Usage in Training

This observation space is defined in the Gymnasium wrapper as:
```python
self.observation_space = gym.spaces.Box(
    low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
)
```

The observation provides the agent with:
- Complete ship state for movement control
- Target information (closest pickup) for navigation
- Collision avoidance data (wall distances in 8 directions)
- Progress tracking (pickups remaining)
- Spatial context (normalized position within level bounds)