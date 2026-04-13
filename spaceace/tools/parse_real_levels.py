#!/usr/bin/env python3
"""
Parse the real SpaceAce level data from mapData.js and convert to Rust format
"""

import re
import json

def parse_js_map_data():
    """Parse the actual JavaScript map data"""
    
    # Read the mapData.js file
    with open('mapData.js', 'r') as f:
        content = f.read()
    
    # Extract the AM array (Level 1 data)
    am_match = re.search(r'const AM = \[(.*?)\];', content, re.DOTALL)
    if not am_match:
        print("Could not find AM array")
        return None
    
    # Parse the array data
    am_data_str = am_match.group(1)
    # Clean up the string and split by commas
    am_data_str = re.sub(r'\s+', ' ', am_data_str)
    am_numbers = [float(x.strip()) for x in am_data_str.split(',') if x.strip()]
    
    print(f"Found {len(am_numbers)} numbers in AM array")
    
    # Parse according to the SpaceAce format
    DL = am_numbers
    
    # Extract data according to JavaScript format
    vertex_count = int(DL[0])
    print(f"Vertex count: {vertex_count}")
    
    # Extract vertices
    vertices = []
    for i in range(vertex_count):
        x = DL[1 + i*2]
        y = DL[1 + i*2 + 1] 
        vertices.append((x, y))
    
    print(f"Extracted {len(vertices)} vertices")
    print(f"First few vertices: {vertices[:5]}")
    
    # Calculate offsets
    OL = 1 + vertex_count * 2 + 1
    PL = int(DL[OL - 1])  # Number of lines
    print(f"Line count: {PL}")
    
    # Extract lines
    lines = []
    for i in range(PL):
        vA = int(DL[OL + i*2])
        vB = int(DL[OL + i*2 + 1])
        lines.append((vA, vB))
    
    print(f"Extracted {len(lines)} lines") 
    print(f"First few lines: {lines[:5]}")
    
    # Ship spawn point
    start_index = int(DL[OL + PL*2])
    print(f"Start index: {start_index}")
    
    # Pickup/bounds data
    QL = OL + PL*2 + 4
    RL = int(DL[QL - 1])  # Number of pickups
    CC = DL[QL - 3]       # Bounding width
    DC = DL[QL - 2]       # Bounding height
    
    print(f"Pickup count: {RL}")
    print(f"Bounding box: {CC} x {DC}")
    
    # Extract pickups
    pickups = []
    for i in range(RL):
        pickup_index = int(DL[QL + i])
        pickups.append(pickup_index)
    
    print(f"Pickup indices: {pickups}")
    
    # Generate Rust code
    rust_level = generate_rust_level_code(vertices, lines, pickups, start_index, CC, DC)
    
    with open('real_level_1.rs', 'w') as f:
        f.write(rust_level)
    
    print("Generated real_level_1.rs")
    
    return {
        'vertices': vertices,
        'lines': lines,
        'pickups': pickups,
        'start_index': start_index,
        'bounding_width': CC,
        'bounding_height': DC
    }

def generate_rust_level_code(vertices, lines, pickups, start_index, width, height):
    """Generate Rust code for the real level"""
    
    code = f"""fn create_real_level_1() -> SpaceAceMapData {{
    // REAL SpaceAce Level 1 data extracted from JavaScript mapData.js
    let vertices = vec![
"""
    
    # Add vertices
    for i, (x, y) in enumerate(vertices):
        code += f"        ({x}, {y}),  // {i}\n"
    
    code += "    ];\n\n"
    code += "    let lines = vec![\n"
    
    # Add lines
    for i, (v1, v2) in enumerate(lines):
        code += f"        ({v1}, {v2}), // Line {i}\n"
    
    code += "    ];\n\n"
    
    code += f"""    SpaceAceMapData {{
        vertices,
        lines,
        pickups: vec!{pickups},  // Pickup vertex indices
        start_index: {start_index},
        bounding_width: {width},
        bounding_height: {height}, 
        triangles: vec![], // Not needed for collision
    }}
}}"""
    
    return code

if __name__ == "__main__":
    level_data = parse_js_map_data()
    if level_data:
        print("Successfully parsed Level 1 data!")
        print(f"Vertices: {len(level_data['vertices'])}")
        print(f"Lines: {len(level_data['lines'])}")
        print(f"Pickups: {len(level_data['pickups'])}")
        print(f"Map size: {level_data['bounding_width']} x {level_data['bounding_height']}")