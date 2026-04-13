# SpaceAce Map Editor

A visual map editor for creating and editing SpaceAce game levels.

## Features

- **Visual Editing**: Click-and-drag interface for creating game maps
- **Multiple Tools**:
  - Add Vertex: Create map vertices (corners/points)
  - Add Line: Connect vertices to create walls
  - Place Pickup: Mark vertices as pickup locations
  - Set Spawn: Define player spawn point
  - Move/Select: Drag vertices to reposition them
  - Delete: Remove vertices, lines, or pickups
- **Level Management**:
  - Load existing levels (0-10)
  - Create new levels from scratch
  - Export levels as JSON
  - Import custom level files
- **View Controls**:
  - Zoom in/out with mouse wheel
  - Pan view (coming soon)
  - Grid display for alignment
- **Real-time Statistics**: Track vertices, lines, pickups, and spawn point

## Usage

1. Open `map_editor.html` in a web browser
2. Select a tool from the sidebar
3. Click on the canvas to perform actions:
   - **Vertex Tool**: Click to add vertices
   - **Line Tool**: Click two vertices to connect them
   - **Pickup Tool**: Click a vertex to toggle pickup status
   - **Spawn Tool**: Click a vertex to set as spawn point
   - **Move Tool**: Drag vertices to new positions
   - **Delete Tool**: Click elements to remove them

## Level Format

The editor exports levels in the SpaceAce format:
```
[vertex_count, x1, y1, x2, y2, ..., line_count, v1a, v1b, v2a, v2b, ..., 
 start_index, map_width, map_height, pickup_count, p1, p2, ...]
```

## Color Legend

- **Red vertices**: Normal map points
- **Yellow vertices**: Pickup locations
- **Green vertex**: Spawn point
- **Magenta vertex**: Currently selected
- **Green cross**: Actual spawn position (100 units above spawn vertex)

## Keyboard Shortcuts

- **Arrow Keys**: Pan the view (move camera)
- **Mouse Wheel**: Zoom in/out
- **Middle Mouse Button**: Click and drag to pan view

## Tips

1. Start by placing vertices at key locations
2. Connect vertices with lines to create walls
3. Place pickups at strategic locations
4. Set the spawn point in a safe area
5. Test your level in the game to ensure playability

## Exporting Levels

1. Click "Export Level Data"
2. The level data will appear in the sidebar and download as a JSON file
3. Add the level to `spaceace_levels.json` to use it in the game

## Level Design Guidelines

- Keep spawn points away from walls (at least 100 units)
- Ensure all pickups are reachable
- Create interesting paths and challenges
- Test collision boundaries carefully
- Consider the ship's physics when designing tight spaces