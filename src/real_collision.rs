use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LineSegment {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl LineSegment {
    pub fn intersects(&self, other: &LineSegment) -> bool {
        lines_intersect(
            (self.x1, self.y1), (self.x2, self.y2),
            (other.x1, other.y1), (other.x2, other.y2)
        )
    }
}

pub struct RealCollisionSystem {
    lines: Vec<LineSegment>,
    // Spatial partitioning grid (exact size from JavaScript)
    grid: HashMap<(i32, i32), Vec<usize>>,
    grid_size: f32, // 500.0 units per cell like JavaScript
}

impl RealCollisionSystem {
    pub fn new() -> Self {
        RealCollisionSystem {
            lines: Vec::new(),
            grid: HashMap::new(),
            grid_size: 500.0, // LINE_GRID_SIZE from JavaScript
        }
    }
    
    pub fn clear(&mut self) {
        self.lines.clear();
        self.grid.clear();
    }
    
    pub fn add_line(&mut self, line: LineSegment) {
        let line_index = self.lines.len();
        self.lines.push(line);
        
        // Add to spatial grid for performance optimization
        let line_ref = &self.lines[line_index];
        let min_x = line_ref.x1.min(line_ref.x2);
        let max_x = line_ref.x1.max(line_ref.x2);
        let min_y = line_ref.y1.min(line_ref.y2);
        let max_y = line_ref.y1.max(line_ref.y2);
        
        let grid_min_x = (min_x / self.grid_size).floor() as i32;
        let grid_max_x = (max_x / self.grid_size).floor() as i32;
        let grid_min_y = (min_y / self.grid_size).floor() as i32;
        let grid_max_y = (max_y / self.grid_size).floor() as i32;
        
        for gx in grid_min_x..=grid_max_x {
            for gy in grid_min_y..=grid_max_y {
                self.grid.entry((gx, gy)).or_insert_with(Vec::new).push(line_index);
            }
        }
    }
    
    pub fn check_collision(&self, ship_segments: &[((f32, f32), (f32, f32))]) -> bool {
        for &((sx1, sy1), (sx2, sy2)) in ship_segments {
            let ship_line = LineSegment {
                x1: sx1, y1: sy1,
                x2: sx2, y2: sy2,
            };
            
            // Get grid cells that this ship segment overlaps
            let min_x = sx1.min(sx2);
            let max_x = sx1.max(sx2);
            let min_y = sy1.min(sy2);
            let max_y = sy1.max(sy2);
            
            let grid_min_x = (min_x / self.grid_size).floor() as i32;
            let grid_max_x = (max_x / self.grid_size).floor() as i32;
            let grid_min_y = (min_y / self.grid_size).floor() as i32;
            let grid_max_y = (max_y / self.grid_size).floor() as i32;
            
            // Check collision with lines in overlapping grid cells
            for gx in grid_min_x..=grid_max_x {
                for gy in grid_min_y..=grid_max_y {
                    if let Some(line_indices) = self.grid.get(&(gx, gy)) {
                        for &line_index in line_indices {
                            if ship_line.intersects(&self.lines[line_index]) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }
    
    pub fn raycast(&self, start_x: f32, start_y: f32, dir_x: f32, dir_y: f32, max_distance: f32) -> f32 {
        let end_x = start_x + dir_x * max_distance;
        let end_y = start_y + dir_y * max_distance;
        
        let mut closest_distance = max_distance;
        
        // Get grid cells along the ray path
        let grid_start_x = (start_x / self.grid_size).floor() as i32;
        let grid_end_x = (end_x / self.grid_size).floor() as i32;
        let grid_start_y = (start_y / self.grid_size).floor() as i32;
        let grid_end_y = (end_y / self.grid_size).floor() as i32;
        
        let grid_min_x = grid_start_x.min(grid_end_x);
        let grid_max_x = grid_start_x.max(grid_end_x);
        let grid_min_y = grid_start_y.min(grid_end_y);
        let grid_max_y = grid_start_y.max(grid_end_y);
        
        for gx in grid_min_x..=grid_max_x {
            for gy in grid_min_y..=grid_max_y {
                if let Some(line_indices) = self.grid.get(&(gx, gy)) {
                    for &line_index in line_indices {
                        let line = &self.lines[line_index];
                        if let Some((ix, iy)) = ray_line_intersection(
                            start_x, start_y, end_x, end_y,
                            line.x1, line.y1, line.x2, line.y2
                        ) {
                            let dx = ix - start_x;
                            let dy = iy - start_y;
                            let distance = (dx * dx + dy * dy).sqrt();
                            if distance < closest_distance {
                                closest_distance = distance;
                            }
                        }
                    }
                }
            }
        }
        
        closest_distance
    }
}

// Exact line intersection algorithm from JavaScript
fn lines_intersect(p1: (f32, f32), p2: (f32, f32), q1: (f32, f32), q2: (f32, f32)) -> bool {
    let s1x = p2.0 - p1.0;
    let s1y = p2.1 - p1.1;
    let s2x = q2.0 - q1.0;
    let s2y = q2.1 - q1.1;
    
    let denom = -s2x * s1y + s1x * s2y;
    if denom.abs() < 0.000001 {
        return false; // Parallel lines
    }
    
    let s = (-s1y * (p1.0 - q1.0) + s1x * (p1.1 - q1.1)) / denom;
    let t = (s2x * (p1.1 - q1.1) - s2y * (p1.0 - q1.0)) / denom;
    
    s >= 0.0 && s <= 1.0 && t >= 0.0 && t <= 1.0
}

// Ray-line intersection for raycasting
fn ray_line_intersection(
    ray_x1: f32, ray_y1: f32, ray_x2: f32, ray_y2: f32,
    line_x1: f32, line_y1: f32, line_x2: f32, line_y2: f32
) -> Option<(f32, f32)> {
    let s1x = ray_x2 - ray_x1;
    let s1y = ray_y2 - ray_y1;
    let s2x = line_x2 - line_x1;
    let s2y = line_y2 - line_y1;
    
    let denom = -s2x * s1y + s1x * s2y;
    if denom.abs() < 0.000001 {
        return None; // Parallel
    }
    
    let s = (-s1y * (ray_x1 - line_x1) + s1x * (ray_y1 - line_y1)) / denom;
    let t = (s2x * (ray_y1 - line_y1) - s2y * (ray_x1 - line_x1)) / denom;
    
    if s >= 0.0 && s <= 1.0 && t >= 0.0 && t <= 1.0 {
        let ix = ray_x1 + t * s1x;
        let iy = ray_y1 + t * s1y;
        Some((ix, iy))
    } else {
        None
    }
}