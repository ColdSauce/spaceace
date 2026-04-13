use std::fs;
use std::collections::HashMap;
use serde_json;

#[derive(Debug, Clone)]
pub struct SpaceAceMapData {
    pub vertices: Vec<(f32, f32)>,
    pub lines: Vec<(usize, usize)>,
    pub pickups: Vec<usize>,  // Indices into vertices array
    pub start_index: usize,
    pub bounding_width: f32,
    pub bounding_height: f32,
    pub triangles: Vec<(usize, usize, usize)>,
}

// Load real SpaceAce level data from JSON file
fn load_spaceace_levels() -> Result<HashMap<usize, Vec<f64>>, Box<dyn std::error::Error>> {
    
    // Try multiple possible paths for the JSON file
    let possible_paths = [
        "data/spaceace_levels.json",
        "./data/spaceace_levels.json",
        "../data/spaceace_levels.json",
        "/Users/coldsauce/projects/spaceace/data/spaceace_levels.json",
    ];
    
    let mut content = String::new();
    let mut found_path = None;
    
    for path in &possible_paths {
        match fs::read_to_string(path) {
            Ok(file_content) => {
                content = file_content;
                found_path = Some(path);
                break;
            }
            Err(_) => {
                continue;
            }
        }
    }
    
    if found_path.is_none() {
        return Err("Could not find spaceace_levels.json file".into());
    }
    
    // Parse JSON - expect format: {"1": [level1_data], "2": [level2_data], ...}
    let json_data: serde_json::Value = serde_json::from_str(&content)?;
    
    let mut result = HashMap::new();
    
    // Extract levels (skip "_raw_arrays" metadata)
    if let serde_json::Value::Object(map) = json_data {
        for (key, value) in map {
            if key.starts_with('_') {
                continue; // Skip metadata
            }
            
            if let Ok(level_num) = key.parse::<usize>() {
                if let serde_json::Value::Array(array) = value {
                    let mut level_data = Vec::new();
                    for item in array {
                        if let serde_json::Value::Number(num) = item {
                            if let Some(f) = num.as_f64() {
                                level_data.push(f);
                            }
                        }
                    }
                    result.insert(level_num, level_data);
                }
            }
        }
    }
    
    Ok(result)
}

// Parse SpaceAce level data format - EXACTLY like JavaScript parseFlashMapData()
fn parse_spaceace_data(data: &[f64]) -> Option<SpaceAceMapData> {
    if data.len() < 2 {
        return None;
    }
    
    let vertex_count = data[0] as usize;
    
    // JavaScript: OL = 1 + vertexCount*2 + 1
    let ol = 1 + vertex_count * 2 + 1;
    
    if data.len() < ol {
        return None;
    }
    
    // JavaScript: PL = DL[OL - 1]; // # lines
    let pl = data[ol - 1] as usize;
    
    // Extract vertices (x, y pairs)
    let mut vertices = Vec::new();
    for i in 0..vertex_count {
        let x = data[1 + i * 2] as f32;
        let y = data[1 + i * 2 + 1] as f32;
        vertices.push((x, y));
    }
    
    // Extract lines - JavaScript: for(let i=0; i<PL; i++){ const vA=DL[OL + i*2]; const vB=DL[OL + i*2 +1]; }
    let mut lines = Vec::new();
    for i in 0..pl {
        let va = data[ol + i * 2] as usize;
        let vb = data[ol + i * 2 + 1] as usize;
        if va < vertex_count && vb < vertex_count {
            lines.push((va, vb));
        }
    }
    
    // JavaScript: startIndex = DL[ OL + PL*2 ]; // The "ship spawn vertex" from the Flash code
    let start_index_offset = ol + pl * 2;
    let start_index = if start_index_offset < data.len() {
        data[start_index_offset] as usize
    } else {
        0
    };
    
    // JavaScript: QL = OL + PL*2 + 4;
    let ql = ol + pl * 2 + 4;
    
    let (rl, cc, dc) = if ql <= data.len() {
        let rl = data.get(ql - 1).unwrap_or(&0.0) as &f64; // # pickups
        let cc = data.get(ql - 3).unwrap_or(&1000.0) as &f64; // bounding width
        let dc = data.get(ql - 2).unwrap_or(&800.0) as &f64; // bounding height
        (*rl as usize, *cc as f32, *dc as f32)
    } else {
        (0, 1000.0, 800.0)
    };
    
    
    // Extract pickups - JavaScript: for(let i=0; i<RL; i++){ pickups.push( DL[QL + i] ); }
    let mut pickups = Vec::new();
    for i in 0..rl {
        if ql + i < data.len() {
            let pickup_vertex = data[ql + i] as usize;
            if pickup_vertex < vertex_count {
                pickups.push(pickup_vertex);
            }
        }
    }
    
    // Extract triangles (if present)
    let tri_offset = ql + rl + 1;
    let mut triangles = Vec::new();
    
    if tri_offset <= data.len() {
        let tri_count = data.get(tri_offset - 1).unwrap_or(&0.0) as &f64;
        let tri_count = *tri_count as usize;
        
        for i in 0..tri_count {
            let t1_idx = tri_offset + i * 3;
            let t2_idx = tri_offset + i * 3 + 1;
            let t3_idx = tri_offset + i * 3 + 2;
            
            if t3_idx < data.len() {
                let t1 = data[t1_idx] as usize;
                let t2 = data[t2_idx] as usize;
                let t3 = data[t3_idx] as usize;
                
                if t1 < vertex_count && t2 < vertex_count && t3 < vertex_count {
                    triangles.push((t1, t2, t3));
                }
            }
        }
    }
    
    Some(SpaceAceMapData {
        vertices,
        lines,
        pickups,
        start_index,
        bounding_width: cc,
        bounding_height: dc,
        triangles,
    })
}

/// Parse a flat JSON array (e.g. `[numVerts, x, y, …]`) into SpaceAceMapData.
/// This is the format produced by `serialize_map` in generate_maps.py.
pub fn parse_map_json(json_str: &str) -> Option<SpaceAceMapData> {
    let arr: Vec<f64> = serde_json::from_str(json_str).ok()?;
    parse_spaceace_data(&arr)
}

pub fn parse_real_map_data(level: usize) -> Option<SpaceAceMapData> {
    
    // Load all level data from JSON file
    let levels_data = match load_spaceace_levels() {
        Ok(data) => {
            data
        },
        Err(e) => {
            return create_fallback_level(level);
        }
    };
    
    // Get data for the requested level
    if let Some(level_data) = levels_data.get(&level) {
        
        // Parse the SpaceAce format data using exact JavaScript logic
        match parse_spaceace_data(level_data) {
            Some(map_data) => {
                Some(map_data)
            }
            None => {
                create_fallback_level(level)
            }
        }
    } else {
        create_fallback_level(level)
    }
}

// Create a fallback level when real data isn't available
fn create_fallback_level(level: usize) -> Option<SpaceAceMapData> {
    
    // Create a simple rectangular level with some pickups
    let width = 1000.0;
    let height = 800.0;
    let margin = 50.0;
    
    let vertices = vec![
        // Boundary rectangle
        (margin, margin),           // 0
        (width - margin, margin),   // 1
        (width - margin, height - margin), // 2
        (margin, height - margin),  // 3
        // Some interior obstacles
        (200.0, 200.0),            // 4
        (300.0, 200.0),            // 5
        (300.0, 300.0),            // 6
        (200.0, 300.0),            // 7
        // Pickup locations
        (150.0, 150.0),            // 8
        (400.0, 400.0),            // 9
        (600.0, 200.0),            // 10
        (700.0, 600.0),            // 11
    ];
    
    let lines = vec![
        // Boundary
        (0, 1), (1, 2), (2, 3), (3, 0),
        // Interior obstacle
        (4, 5), (5, 6), (6, 7), (7, 4),
    ];
    
    let pickups = vec![8, 9, 10, 11];
    
    Some(SpaceAceMapData {
        vertices,
        lines,
        pickups,
        start_index: 0,
        bounding_width: width,
        bounding_height: height,
        triangles: Vec::new(),
    })
}