# SpaceAce Level Format Documentation

## Overview

The SpaceAce level system uses a compact, sequential binary format originally designed for Adobe Flash. Level data is stored as flat arrays of numbers that are parsed sequentially to extract game geometry, pickup locations, and spawn points.

## Level Data Sources

### Primary Data Files
- **`spaceace_levels.json`**: Contains 10 levels in parsed format (86,735 lines total)
- **`mapData.js`**: Original JavaScript arrays (`AM`, `BM`, `CM`, etc.) from Flash game
- **`map_data.json`**: Alternative JSON representation

### Available Levels
The game contains **11 levels** (including custom Level 0) with varying complexity:

| Level | Vertices | Lines | Pickups | Map Size | Complexity | Difficulty |
|-------|----------|-------|---------|----------|------------|------------|
| **0** | **6** | **4** | **1** | **800×600** | **Ultra-Low** | **Ultra Easy** (training level) |
| **1** | 199 | 185 | 17 | 2485×2294 | High | Hard (original training level) |
| **2** | 188 | 179 | 9 | 2338×3068 | High | Hard |
| **3** | 276 | 266 | 9 | 3947×4402 | **Highest** | Extreme (largest map) |
| **4** | 144 | 130 | 13 | 2327×3249 | Medium | Medium |
| **5** | 173 | 168 | 7 | 2442×3331 | Medium | Medium |
| **6** | 139 | 131 | 7 | 1868×2048 | Low | Easy (smallest original map) |
| **7** | 161 | 157 | 3 | 1750×3234 | Low | Easy (fewest pickups) |
| **8** | 85 | 73 | 11 | 2076×1887 | Low | Easy (fewest vertices) |
| **9** | 176 | 158 | 17 | 2221×3491 | High | Hard (most pickups) |
| **10** | 216 | 210 | 7 | 4211×2841 | High | Hard (widest map) |

## Binary Data Format

The level data follows a strict sequential format where each section must be read in order:

### Format Structure
```
[vertex_count, x1, y1, x2, y2, ..., line_count, v1a, v1b, v2a, v2b, ..., 
 start_index, padding, bounding_width, bounding_height, pickup_count, p1, p2, ..., 
 triangle_count, t1a, t1b, t1c, ...]
```

### Parsing Algorithm (from `real_map_parser.rs`)

#### Step 1: Read Vertex Count
```rust
let vertex_count = data[0] as usize;
```

#### Step 2: Extract Vertices
```rust
// Vertices start at index 1, each vertex is (x, y)
let mut vertices = Vec::new();
for i in 0..vertex_count {
    let x = data[1 + i * 2] as f32;
    let y = data[1 + i * 2 + 1] as f32;
    vertices.push((x, y));
}
```

#### Step 3: Calculate Line Data Offset
```rust
// OL = 1 (for vertex_count) + vertex_count*2 (for vertex coordinates) + 1
let ol = 1 + vertex_count * 2 + 1;
let line_count = data[ol - 1] as usize;  // Number of lines
```

#### Step 4: Extract Lines
```rust
// Lines are pairs of vertex indices
let mut lines = Vec::new();
for i in 0..line_count {
    let va = data[ol + i * 2] as usize;      // First vertex index
    let vb = data[ol + i * 2 + 1] as usize;  // Second vertex index
    if va < vertex_count && vb < vertex_count {
        lines.push((va, vb));
    }
}
```

#### Step 5: Extract Metadata
```rust
// Ship spawn vertex index
let start_index_offset = ol + line_count * 2;
let start_index = data[start_index_offset] as usize;

// Bounding box and pickup offset
let ql = ol + line_count * 2 + 4;
let bounding_width = data[ql - 3] as f32;
let bounding_height = data[ql - 2] as f32;
let pickup_count = data[ql - 1] as usize;
```

#### Step 6: Extract Pickups
```rust
// Pickups are vertex indices where collectibles are placed
let mut pickups = Vec::new();
for i in 0..pickup_count {
    if ql + i < data.len() {
        let pickup_vertex = data[ql + i] as usize;
        if pickup_vertex < vertex_count {
            pickups.push(pickup_vertex);
        }
    }
}
```

#### Step 7: Extract Triangles (Optional)
```rust
let tri_offset = ql + pickup_count + 1;
let triangle_count = data[tri_offset - 1] as usize;

// Each triangle is defined by 3 vertex indices
for i in 0..triangle_count {
    let t1 = data[tri_offset + i * 3] as usize;
    let t2 = data[tri_offset + i * 3 + 1] as usize;
    let t3 = data[tri_offset + i * 3 + 2] as usize;
    triangles.push((t1, t2, t3));
}
```

## Data Structure Components

### Vertices
- **Format**: `(x: f32, y: f32)` coordinate pairs
- **Purpose**: Define all geometry points in the level
- **Coordinate System**: Pixel coordinates, origin varies by level

### Lines
- **Format**: `(vertex_index_a: usize, vertex_index_b: usize)` pairs
- **Purpose**: Connect vertices to form walls and collision boundaries
- **Usage**: Used by collision system for wall detection

### Pickups
- **Format**: `vertex_index: usize` (references vertices array)
- **Purpose**: Define locations of collectible items
- **Game Logic**: Ship must collect all pickups to complete level

### Start Index
- **Format**: `vertex_index: usize`
- **Purpose**: Defines where ship spawns
- **Implementation**: Ship spawns 100 units above the specified vertex (see `real_physics.rs:48-50`)

### Bounding Box
- **Format**: `(width: f32, height: f32)`
- **Purpose**: Defines level boundaries for normalization and bounds checking

## Level Loading System

### Rust Implementation (`real_map_parser.rs`)
```rust
pub fn parse_real_map_data(level: usize) -> Option<SpaceAceMapData> {
    // 1. Load JSON data from spaceace_levels.json
    // 2. Extract array for requested level
    // 3. Parse using exact JavaScript logic
    // 4. Return structured SpaceAceMapData
}
```

### Level Selection Logic
```rust
// Multiple fallback paths for JSON file
let possible_paths = [
    "spaceace_levels.json",
    "./spaceace_levels.json", 
    "../spaceace_levels.json",
    "/Users/coldsauce/projects/spaceace/spaceace_levels.json",
];
```

### Fallback System
If level data cannot be loaded, the system creates a simple rectangular test level with:
- 12 vertices forming a boundary and obstacle
- 8 lines creating walls
- 4 pickups at strategic locations
- 1000×800 pixel bounds

## Integration with Game Engine

### Physics Integration
- **Ship Spawn**: `self.physics.reset(spawn_x, spawn_y - 100.0)` (100 units above start vertex)
- **Collision**: Lines converted to `LineSegment` objects for wall collision detection
- **Pickup Collection**: Distance-based collision with 10-unit pickup radius

### Observation Space
The level data feeds into the RL observation space:
- **Wall distances**: 8-direction raycasting using line geometry
- **Pickup locations**: Closest pickup position and distance
- **Map bounds**: Used for position normalization [0,1]

### Rendering
- **ASCII Mode**: Simplified representation for debugging
- **Detailed Mode**: Full geometry visualization
- **Map lines**: Rendered as wall segments
- **Pickups**: Rendered as circles at vertex positions

## Usage for RL Training

## Level 0 - Custom Training Level

### Design Philosophy
Level 0 was specifically created to solve the RL training problem. The original levels were too complex for initial learning, so Level 0 provides:

- **Minimal Geometry**: Just 4 boundary walls (no internal obstacles)
- **Single Objective**: Only 1 pickup to collect
- **Safe Spawn**: Ship starts away from walls at (50, 150)
- **Clear Goal**: Pickup at (300, 150) requires simple right movement
- **Small Scale**: 400×300 map fits easily in agent's perception

### Level 0 Data Structure
```json
{
  "0": [
    5,                          // 5 vertices
    50, 50,                     // Vertex 0: bottom-left boundary
    350, 50,                    // Vertex 1: bottom-right boundary  
    350, 250,                   // Vertex 2: top-right boundary
    50, 250,                    // Vertex 3: top-left boundary
    300, 150,                   // Vertex 4: pickup location
    4,                          // 4 lines
    0, 1, 1, 2, 2, 3, 3, 0,    // Boundary lines forming rectangle
    3,                          // Start at vertex 3 (ship spawns at 50, 150)
    400.0, 300.0,              // Map bounds
    1,                          // 1 pickup
    4                           // Pickup at vertex 4
  ]
}
```

### Curriculum Learning Recommendations
**NEW RECOMMENDED PROGRESSION** based on complexity analysis:

1. **Start Ultra-Easy**: **Level 0** (custom training level)
2. **Progress Easy**: Level 8 (85 vertices) → Level 6 → Level 7
3. **Progress Medium**: Level 5 → Level 4
4. **Advanced**: Level 2 → Level 9 → Level 1 → Level 10 → Level 3

### Training Issue Resolution
**Level 1 was the problem!** With 199 vertices, 185 lines, and 17 pickups, it's one of the most complex levels. Level 0 solves this by being **17× simpler** in every dimension.

### Level Selection for Training
```python
# Start with ultra-simple custom level
beginner_level = [0]           # Custom ultra-simple level

# Easy levels for progression
easy_levels = [8, 6, 7]        # Simple geometry, fewer pickups

# Medium difficulty  
medium_levels = [5, 4]         # Moderate complexity

# Hard levels
hard_levels = [2, 9, 1, 10, 3] # Complex geometry, many pickups
```

## File Relationships

- **`spaceace_levels.json`**: Primary data source (parsed format)
- **`real_map_parser.rs`**: Rust parser implementing JavaScript logic
- **`real_game.rs`**: Game engine consuming parsed data
- **`real_collision.rs`**: Collision system using line geometry
- **`spaceace_env.py`**: RL environment interface

## Historical Context

This format preserves the exact data structure from the original Adobe Flash SpaceAce game, maintaining pixel-perfect compatibility with the original level designs while enabling modern RL training and analysis.