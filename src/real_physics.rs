use std::f32::consts::PI;

// Exact physics constants from JavaScript SpaceAce
const GRAVITY: f32 = 100.0;
const THRUST_POWER: f32 = 400.0;
const ROTATION_SPEED: f32 = 4.363323;

// Ship geometry constants
const PICKUP_RADIUS: f32 = 10.0;
const SHIP_COLLISION_RADIUS: f32 = 36.5;

#[derive(Debug, Clone)]
pub struct RealPhysicsState {
    // Ship position and velocity
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub rotation: f32,
    
    // Control inputs
    pub rotating_left: bool,
    pub rotating_right: bool,
    pub thrusting: bool,
    pub exploded: bool,
    
    // Performance optimization
    collision_skip_frames: u32,
}

impl RealPhysicsState {
    pub fn new() -> Self {
        RealPhysicsState {
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            rotation: 0.0,
            rotating_left: false,
            rotating_right: false,
            thrusting: false,
            exploded: false,
            collision_skip_frames: 0,
        }
    }
    
    pub fn reset(&mut self, start_x: f32, start_y: f32) {
        // Spawn position is already offset by the caller (real_game.rs)
        self.x = start_x;
        self.y = start_y;
        self.vx = 0.0;
        self.vy = 0.0;
        self.rotation = 0.0;
        self.rotating_left = false;
        self.rotating_right = false;
        self.thrusting = false;
        self.exploded = false;
        self.collision_skip_frames = 0;
    }
    
    pub fn set_controls(&mut self, rotate_left: bool, rotate_right: bool, thrust: bool) {
        self.rotating_left = rotate_left;
        self.rotating_right = rotate_right;
        self.thrusting = thrust;
    }
    
    pub fn update(&mut self, dt: f32) {
        if self.exploded {
            return;
        }
        
        // Clamp timestep for stability (max 30fps like JavaScript)
        let clamped_dt = dt.min(1.0 / 30.0);
        
        // Rotation
        if self.rotating_left {
            self.rotation -= ROTATION_SPEED * clamped_dt;
        }
        if self.rotating_right {
            self.rotation += ROTATION_SPEED * clamped_dt;
        }
        
        // Thrust (angle offset by -PI/2 because 0° = up in SpaceAce)
        if self.thrusting {
            let angle = self.rotation - PI * 0.5;
            self.vx += THRUST_POWER * angle.cos() * clamped_dt;
            self.vy += THRUST_POWER * angle.sin() * clamped_dt;
        }
        
        // Gravity
        self.vy += GRAVITY * clamped_dt;
        
        // Position integration
        self.x += self.vx * clamped_dt;
        self.y += self.vy * clamped_dt;
    }
    
    pub fn get_position(&self) -> (f32, f32) {
        (self.x, self.y)
    }
    
    pub fn get_velocity(&self) -> (f32, f32) {
        (self.vx, self.vy)
    }
    
    pub fn get_rotation(&self) -> f32 {
        self.rotation
    }
    
    pub fn is_thrusting(&self) -> bool {
        self.thrusting
    }
    
    pub fn is_rotating_left(&self) -> bool {
        self.rotating_left
    }
    
    pub fn is_rotating_right(&self) -> bool {
        self.rotating_right
    }
    
    pub fn should_skip_collision(&mut self) -> bool {
        // Performance optimization from JavaScript:
        // Skip collision when moving very fast (> 100000 speed²)
        let speed_squared = self.vx * self.vx + self.vy * self.vy;
        if speed_squared > 100000.0 {
            self.collision_skip_frames += 1;
            return self.collision_skip_frames % 2 != 0; // Skip every other frame
        }
        false
    }
    
    pub fn explode(&mut self) {
        self.exploded = true;
    }
    
    // Get exact ship collision geometry (from JavaScript shipVerts)
    pub fn get_ship_collision_segments(&self) -> Vec<((f32, f32), (f32, f32))> {
        // Exact ship vertices from JavaScript
        let ship_verts = [
            (0.0, -36.5),      // 0: nose tip
            (-19.0, 23.5),     // 1: left rear
            (-24.0, 23.5),     // 2: left rear outer
            (-15.675, 13.0),   // 3: left wing
            (19.0, 23.5),      // 4: right rear  
            (24.0, 23.5),      // 5: right rear outer
            (15.675, 13.0),    // 6: right wing
            (0.0, 67.45),      // 7: thruster center
            (-14.1075, 13.0),  // 8: left thruster
            (14.1075, 13.0),   // 9: right thruster
        ];
        
        // Transform vertices by ship position and rotation
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        
        let mut transformed_verts = Vec::new();
        for &(vx, vy) in &ship_verts {
            let tx = vx * cos_r - vy * sin_r + self.x;
            let ty = vx * sin_r + vy * cos_r + self.y;
            transformed_verts.push((tx, ty));
        }
        
        // Exact collision segments from JavaScript
        vec![
            (transformed_verts[3], transformed_verts[6]), // Wing span line
            (transformed_verts[2], transformed_verts[1]), // Left rear
            (transformed_verts[1], transformed_verts[0]), // Left side to nose
            (transformed_verts[0], transformed_verts[4]), // Right side from nose
            (transformed_verts[4], transformed_verts[5]), // Right rear
        ]
    }
    
    // Get ship vertices for rendering
    pub fn get_ship_render_vertices(&self) -> Vec<(f32, f32)> {
        // Simplified triangle for rendering (nose, left wing, right wing)
        let render_verts = [
            (0.0, -36.5),      // Nose
            (-15.675, 13.0),   // Left wing
            (15.675, 13.0),    // Right wing
        ];
        
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        
        render_verts.iter().map(|&(vx, vy)| {
            let tx = vx * cos_r - vy * sin_r + self.x;
            let ty = vx * sin_r + vy * cos_r + self.y;
            (tx, ty)
        }).collect()
    }
    
    // Get thrust effect vertices for rendering
    pub fn get_thrust_vertices(&self) -> Vec<(f32, f32)> {
        if !self.thrusting {
            return vec![];
        }
        
        // Thruster vertices (from ship geometry)
        let thrust_verts = [
            (-14.1075, 13.0),  // Left thruster
            (0.0, 67.45),      // Thruster center (flame tip)
            (14.1075, 13.0),   // Right thruster
        ];
        
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        
        thrust_verts.iter().map(|&(vx, vy)| {
            let tx = vx * cos_r - vy * sin_r + self.x;
            let ty = vx * sin_r + vy * cos_r + self.y;
            (tx, ty)
        }).collect()
    }
    
    // Check pickup collision (exact radius from JavaScript)
    pub fn check_pickup_collision(&self, pickup_x: f32, pickup_y: f32) -> bool {
        let radius_squared = (SHIP_COLLISION_RADIUS + PICKUP_RADIUS) * (SHIP_COLLISION_RADIUS + PICKUP_RADIUS);
        let dx = self.x - pickup_x;
        let dy = self.y - pickup_y;
        let dist_squared = dx * dx + dy * dy;
        dist_squared <= radius_squared
    }
}