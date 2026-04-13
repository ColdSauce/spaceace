use crate::real_physics::RealPhysicsState;
use crate::real_collision::{RealCollisionSystem, LineSegment};
use crate::real_map_parser::{SpaceAceMapData, parse_real_map_data};

#[derive(Debug, Clone)]
pub struct RealGameState {
    pub ship_x: f32,
    pub ship_y: f32,
    pub ship_vx: f32,
    pub ship_vy: f32,
    pub ship_rotation: f32,
    pub pickups_remaining: i32,
    pub game_time: f32,
    pub ship_exploded: bool,
    pub level_completed: bool,
}

#[derive(Debug, Clone)]
pub struct RealMapBounds {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
}

#[derive(Debug, Clone)]
pub struct RealPickup {
    pub x: f32,
    pub y: f32,
    pub collected: bool,
}

#[derive(Debug, Clone)]
pub struct GameSnapshot {
    pub physics: RealPhysicsState,
    pub pickups: Vec<RealPickup>,
    pub pickups_collected_this_step: i32,
    pub game_time: f32,
}

pub struct RealSpaceAceGame {
    physics: RealPhysicsState,
    collision_system: RealCollisionSystem,
    map_data: Option<SpaceAceMapData>,
    pickups: Vec<RealPickup>,
    pickups_collected_this_step: i32,
    game_time: f32,
    current_level: usize,
}

impl RealSpaceAceGame {
    pub fn new() -> Self {
        RealSpaceAceGame {
            physics: RealPhysicsState::new(),
            collision_system: RealCollisionSystem::new(),
            map_data: None,
            pickups: Vec::new(),
            pickups_collected_this_step: 0,
            game_time: 0.0,
            current_level: 1,
        }
    }

    pub fn load_level(&mut self, level: usize) -> Result<(), String> {
        self.current_level = level;
        
        // Load real SpaceAce level data
        if let Some(map_data) = parse_real_map_data(level) {
            // Setup collision system with exact map geometry
            self.collision_system.clear();
            for &(i1, i2) in &map_data.lines {
                let p1 = map_data.vertices[i1];
                let p2 = map_data.vertices[i2];
                self.collision_system.add_line(LineSegment {
                    x1: p1.0,
                    y1: p1.1,
                    x2: p2.0,
                    y2: p2.1,
                });
            }
            
            // Setup pickups from map data
            self.pickups.clear();
            for &pickup_index in &map_data.pickups {
                let pickup_vertex = map_data.vertices[pickup_index];
                self.pickups.push(RealPickup {
                    x: pickup_vertex.0,
                    y: pickup_vertex.1,
                    collected: false,
                });
            }
            
            self.map_data = Some(map_data);
        } else {
            return Err(format!("Failed to load level {}", level));
        }
        
        self.reset();
        Ok(())
    }

    pub fn reset(&mut self) {
        
        // Use proper spawn logic matching JavaScript implementation
        let (spawn_x, spawn_y) = if let Some(ref map_data) = self.map_data {
            // JavaScript logic: use startIndex to get spawn vertex, then place ship slightly above
            if map_data.start_index < map_data.vertices.len() {
                let spawn_vertex = map_data.vertices[map_data.start_index];
                let spawn_x = spawn_vertex.0;
                let spawn_y = spawn_vertex.1 - 100.0; // place slightly above the exact spot
                (spawn_x, spawn_y)
            } else {
                // Fallback: center of map bounds + 200 offset from bottom
                let bounds = self.get_map_bounds();
                let fallback_x = (bounds.min_x + bounds.max_x) * 0.5;
                let fallback_y = bounds.min_y + 200.0;
                (fallback_x, fallback_y)
            }
        } else {
            // No map data available - use center position
            (600.0, 400.0)
        };
        
        self.physics.reset(spawn_x, spawn_y);
        
        // Reset pickups
        for pickup in &mut self.pickups {
            pickup.collected = false;
        }
        
        self.game_time = 0.0;
        self.pickups_collected_this_step = 0;
    }

    pub fn set_controls(&mut self, rotate_left: bool, rotate_right: bool, thrust: bool) {
        self.physics.set_controls(rotate_left, rotate_right, thrust);
    }

    pub fn step(&mut self, dt: f32) {
        if self.physics.exploded || self.is_level_completed() {
            return;
        }
        
        self.pickups_collected_this_step = 0;
        
        // Update physics with exact SpaceAce mechanics
        self.physics.update(dt);
        self.game_time += dt;
        
        // Check collisions with walls (with performance optimization)
        if !self.physics.should_skip_collision() {
            let ship_segments = self.physics.get_ship_collision_segments();
            if self.collision_system.check_collision(&ship_segments) {
                self.physics.explode();
                return;
            }
        }
        
        // Check pickup collection with exact radius
        for pickup in &mut self.pickups {
            if !pickup.collected {
                if self.physics.check_pickup_collision(pickup.x, pickup.y) {
                    pickup.collected = true;
                    self.pickups_collected_this_step += 1;
                }
            }
        }
        
        // Check win condition
        if self.get_pickups_remaining() == 0 {
            // Level completed! 
            // In real SpaceAce, this would trigger endRun() and save times
        }
    }

    pub fn get_state(&self) -> RealGameState {
        let pos = self.physics.get_position();
        let vel = self.physics.get_velocity();
        let rotation = self.physics.get_rotation();
        
        RealGameState {
            ship_x: pos.0,
            ship_y: pos.1,
            ship_vx: vel.0,
            ship_vy: vel.1,
            ship_rotation: rotation,
            pickups_remaining: self.get_pickups_remaining(),
            game_time: self.game_time,
            ship_exploded: self.physics.exploded,
            level_completed: self.is_level_completed(),
        }
    }

    pub fn get_pickups_remaining(&self) -> i32 {
        self.pickups.iter().filter(|p| !p.collected).count() as i32
    }

    pub fn get_pickups_collected_this_step(&self) -> i32 {
        self.pickups_collected_this_step
    }

    pub fn is_terminated(&self) -> bool {
        self.physics.exploded || self.is_level_completed()
    }

    pub fn is_ship_exploded(&self) -> bool {
        self.physics.exploded
    }

    pub fn is_level_completed(&self) -> bool {
        self.get_pickups_remaining() == 0
    }

    pub fn get_closest_pickup(&self) -> (f32, f32, f32) {
        let ship_pos = self.physics.get_position();
        let mut closest_dist = f32::MAX;
        let mut closest_x = 0.0;
        let mut closest_y = 0.0;
        
        for pickup in &self.pickups {
            if !pickup.collected {
                let dx = pickup.x - ship_pos.0;
                let dy = pickup.y - ship_pos.1;
                let dist = (dx * dx + dy * dy).sqrt();
                
                if dist < closest_dist {
                    closest_dist = dist;
                    closest_x = pickup.x;
                    closest_y = pickup.y;
                }
            }
        }
        
        if closest_dist == f32::MAX {
            // No pickups remaining
            (ship_pos.0, ship_pos.1, 0.0)
        } else {
            (closest_x, closest_y, closest_dist)
        }
    }

    pub fn get_wall_distances(&self) -> [f32; 8] {
        let ship_pos = self.physics.get_position();
        let ship_rotation = self.physics.get_rotation();
        let mut distances = [1000.0; 8]; // Max distance if no wall found

        // 8 directions relative to ship heading:
        // Forward, Forward-Right, Right, Back-Right, Back, Back-Left, Left, Forward-Left
        // In ship-local space, "forward" = (0, -1) (up), matching rotation=0 convention
        let base_directions = [
            (0.0_f32, -1.0),      // Forward
            (0.707, -0.707),      // Forward-Right
            (1.0, 0.0),          // Right
            (0.707, 0.707),      // Back-Right
            (0.0, 1.0),          // Back
            (-0.707, 0.707),     // Back-Left
            (-1.0, 0.0),         // Left
            (-0.707, -0.707),    // Forward-Left
        ];

        let cos_r = ship_rotation.cos();
        let sin_r = ship_rotation.sin();

        for (i, &(dx, dy)) in base_directions.iter().enumerate() {
            // Rotate direction by ship rotation into world space
            let world_dx = dx * cos_r - dy * sin_r;
            let world_dy = dx * sin_r + dy * cos_r;
            distances[i] = self.collision_system.raycast(ship_pos.0, ship_pos.1, world_dx, world_dy, 1000.0);
        }

        distances
    }

    pub fn get_map_bounds(&self) -> RealMapBounds {
        if let Some(ref map) = self.map_data {
            let mut min_x = f32::MAX;
            let mut max_x = f32::MIN;
            let mut min_y = f32::MAX;
            let mut max_y = f32::MIN;
            for &(x, y) in &map.vertices {
                if x < min_x { min_x = x; }
                if x > max_x { max_x = x; }
                if y < min_y { min_y = y; }
                if y > max_y { max_y = y; }
            }
            // Add margin for ship size
            let margin = 50.0;
            RealMapBounds {
                min_x: min_x - margin,
                max_x: max_x + margin,
                min_y: min_y - margin,
                max_y: max_y + margin,
            }
        } else {
            RealMapBounds {
                min_x: 0.0,
                max_x: 1200.0,
                min_y: 0.0,
                max_y: 800.0,
            }
        }
    }

    pub fn get_ship_render_data(&self) -> (Vec<(f32, f32)>, Vec<(f32, f32)>) {
        let ship_vertices = self.physics.get_ship_render_vertices();
        let thrust_vertices = self.physics.get_thrust_vertices();
        (ship_vertices, thrust_vertices)
    }

    pub fn get_pickup_positions(&self) -> Vec<(f32, f32, bool)> {
        self.pickups.iter().map(|p| (p.x, p.y, p.collected)).collect()
    }

    pub fn get_map_lines(&self) -> Vec<(f32, f32, f32, f32)> {
        if let Some(ref map) = self.map_data {
            map.lines.iter().map(|&(i1, i2)| {
                let p1 = map.vertices[i1];
                let p2 = map.vertices[i2];
                (p1.0, p1.1, p2.0, p2.1)
            }).collect()
        } else {
            vec![]
        }
    }

    pub fn get_map_geometry(&self) -> (Vec<(f32, f32)>, Vec<(usize, usize)>) {
        if let Some(ref map) = self.map_data {
            (map.vertices.clone(), map.lines.clone())
        } else {
            (vec![], vec![])
        }
    }

    pub fn save_state(&self) -> GameSnapshot {
        GameSnapshot {
            physics: self.physics.clone(),
            pickups: self.pickups.clone(),
            pickups_collected_this_step: self.pickups_collected_this_step,
            game_time: self.game_time,
        }
    }

    pub fn load_state(&mut self, snapshot: GameSnapshot) {
        self.physics = snapshot.physics;
        self.pickups = snapshot.pickups;
        self.pickups_collected_this_step = snapshot.pickups_collected_this_step;
        self.game_time = snapshot.game_time;
    }

    pub fn render_ascii(&self) -> String {
        let state = self.get_state();
        let bounds = self.get_map_bounds();
        
        // Create ASCII representation
        let grid_width = 80;
        let grid_height = 25;
        let mut grid = vec![vec![' '; grid_width]; grid_height];
        
        // Draw boundaries
        for x in 0..grid_width {
            grid[0][x] = '#';  // Top
            grid[grid_height-1][x] = '#';  // Bottom
        }
        for y in 0..grid_height {
            grid[y][0] = '#';  // Left
            grid[y][grid_width-1] = '#';  // Right
        }
        
        // Draw map lines (sample some for ASCII)
        let map_lines = self.get_map_lines();
        for (x1, y1, _x2, _y2) in map_lines.iter().take(20) { // Limit for ASCII
            let grid_x1 = ((x1 - bounds.min_x) / (bounds.max_x - bounds.min_x) * (grid_width as f32 - 2.0) + 1.0) as usize;
            let grid_y1 = ((y1 - bounds.min_y) / (bounds.max_y - bounds.min_y) * (grid_height as f32 - 2.0) + 1.0) as usize;
            
            if grid_x1 < grid_width && grid_y1 < grid_height {
                grid[grid_y1][grid_x1] = '▬';
            }
        }
        
        // Draw pickups
        for pickup in &self.pickups {
            if !pickup.collected {
                let grid_x = ((pickup.x - bounds.min_x) / (bounds.max_x - bounds.min_x) * (grid_width as f32 - 2.0) + 1.0) as usize;
                let grid_y = ((pickup.y - bounds.min_y) / (bounds.max_y - bounds.min_y) * (grid_height as f32 - 2.0) + 1.0) as usize;
                
                if grid_x < grid_width && grid_y < grid_height {
                    grid[grid_y][grid_x] = '●';
                }
            }
        }
        
        // Draw ship
        let ship_grid_x = ((state.ship_x - bounds.min_x) / (bounds.max_x - bounds.min_x) * (grid_width as f32 - 2.0) + 1.0) as usize;
        let ship_grid_y = ((state.ship_y - bounds.min_y) / (bounds.max_y - bounds.min_y) * (grid_height as f32 - 2.0) + 1.0) as usize;
        
        if ship_grid_x < grid_width && ship_grid_y < grid_height {
            let ship_char = if state.ship_exploded { '✕' }
                          else if self.physics.is_thrusting() { '▲' }  
                          else { '△' };
            grid[ship_grid_y][ship_grid_x] = ship_char;
        }
        
        // Convert grid to string
        let mut result = String::new();
        result.push_str(&format!("SpaceAce Level {} - Time: {:.1}s\n", self.current_level, state.game_time));
        
        for row in &grid {
            for &cell in row {
                result.push(cell);
            }
            result.push('\n');
        }
        
        result.push_str(&format!(
            "Ship: ({:.0}, {:.0}) vel=({:.1}, {:.1}) rot={:.2}\n\
            Pickups: {} | Status: {}\n\
            Legend: △=Ship, ▲=Thrusting, ✕=Crashed, ●=Pickup, ▬=Wall",
            state.ship_x, state.ship_y,
            state.ship_vx, state.ship_vy,
            state.ship_rotation,
            state.pickups_remaining,
            if state.ship_exploded { "CRASHED" }
            else if state.level_completed { "COMPLETED" }
            else { "ACTIVE" }
        ));
        
        result
    }

    pub fn render_detailed(&self) -> String {
        let state = self.get_state();
        let (closest_pickup_x, closest_pickup_y, closest_pickup_dist) = self.get_closest_pickup();
        let wall_distances = self.get_wall_distances();
        
        format!(
            "=== SpaceAce Detailed State (Real Physics) ===\n\
            Ship Position: ({:.1}, {:.1})\n\
            Ship Velocity: ({:.1}, {:.1}) speed={:.1}\n\
            Ship Rotation: {:.2} rad ({:.0}°)\n\
            Controls: L={} R={} T={}\n\
            \n\
            Closest Pickup: ({:.1}, {:.1}) dist={:.1}\n\
            Pickups Remaining: {}\n\
            \n\
            Wall Distances:\n\
            N:{:.0} NE:{:.0} E:{:.0} SE:{:.0}\n\
            S:{:.0} SW:{:.0} W:{:.0} NW:{:.0}\n\
            \n\
            Game Time: {:.2}s\n\
            Level: {}\n\
            Status: {}\n",
            state.ship_x, state.ship_y,
            state.ship_vx, state.ship_vy, 
            (state.ship_vx * state.ship_vx + state.ship_vy * state.ship_vy).sqrt(),
            state.ship_rotation, state.ship_rotation * 180.0 / std::f32::consts::PI,
            self.physics.is_rotating_left(), self.physics.is_rotating_right(), self.physics.is_thrusting(),
            closest_pickup_x, closest_pickup_y, closest_pickup_dist,
            state.pickups_remaining,
            wall_distances[0], wall_distances[1], wall_distances[2], wall_distances[3],
            wall_distances[4], wall_distances[5], wall_distances[6], wall_distances[7],
            state.game_time,
            self.current_level,
            if state.ship_exploded { "CRASHED" }
            else if state.level_completed { "COMPLETED" }
            else { "ACTIVE" }
        )
    }
}