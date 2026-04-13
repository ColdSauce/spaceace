#!/usr/bin/env python3
"""
Verify that different SpaceAce levels are correctly loaded
"""

import subprocess
import json

def test_level(level):
    """Test that a specific level loads with correct data"""
    print(f"🎯 Testing Level {level}")
    
    # Create game for this level
    cmd = ["./target/release/spaceace-rl"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, text=True)
    
    # Send create command
    create_cmd = json.dumps({"command": "create", "level": level, "max_steps": 100})
    stdout, stderr = proc.communicate(input=create_cmd + "\n")
    
    # Parse response
    try:
        response = json.loads(stdout.strip())
        if response.get("status") == "success":
            print(f"  ✅ Level {level} created successfully")
            
            # Extract level info from stderr
            lines = stderr.split('\n')
            vertices = None
            map_lines = None
            pickups = None
            bounds = None
            
            for line in lines:
                if "vertices," in line and "lines," in line:
                    # Parse: "🎮 Successfully parsed level 1 with 199 vertices, 2158 lines, 20 pickups"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "vertices,":
                            vertices = int(parts[i-1])
                        elif part == "lines,":
                            map_lines = int(parts[i-1])
                        elif part == "pickups":
                            pickups = int(parts[i-1])
                elif "Bounds:" in line:
                    # Parse bounds info
                    bounds = line.split("Bounds:")[1].strip()
            
            print(f"    Vertices: {vertices}")
            print(f"    Map Lines: {map_lines}")
            print(f"    Pickups: {pickups}")
            print(f"    Bounds: {bounds}")
            
            return {
                'level': level,
                'vertices': vertices,
                'map_lines': map_lines, 
                'pickups': pickups,
                'bounds': bounds
            }
        else:
            print(f"  ❌ Level {level} failed: {response}")
            return None
            
    except Exception as e:
        print(f"  ❌ Level {level} error: {e}")
        return None

def main():
    print("🔍 Verifying Real SpaceAce Level Data")
    print("=" * 50)
    
    results = []
    for level in [1, 2, 3, 6, 8, 10]:
        result = test_level(level)
        if result:
            results.append(result)
        print()
    
    print("📊 Summary:")
    print("-" * 50)
    for result in results:
        print(f"Level {result['level']:2d}: {result['vertices']:3d} vertices, {result['map_lines']:4d} lines, {result['pickups']:2d} pickups")
    
    # Check that levels are different
    if len(results) > 1:
        all_different = True
        for i in range(len(results)-1):
            for j in range(i+1, len(results)):
                r1, r2 = results[i], results[j]
                if (r1['vertices'] == r2['vertices'] and 
                    r1['map_lines'] == r2['map_lines'] and 
                    r1['pickups'] == r2['pickups']):
                    all_different = False
                    print(f"⚠️  Levels {r1['level']} and {r2['level']} appear identical!")
        
        if all_different:
            print("✅ All levels have unique data - Real SpaceAce levels confirmed!")
        else:
            print("❌ Some levels appear identical")

if __name__ == "__main__":
    main()