#!/usr/bin/env python3
"""
Convert mapData.js to JSON format for Rust consumption
"""

import json
import re
import sys

def extract_map_data(js_file_path):
    """Extract all level arrays from mapData.js"""
    with open(js_file_path, 'r') as f:
        content = f.read()
    
    # Find all const XM = [...] patterns
    level_pattern = r'const ([A-J]M) = \[(.*?)\];'
    matches = re.findall(level_pattern, content, re.DOTALL)
    
    levels = {}
    
    for level_name, array_content in matches:
        # Clean up the array content and split by commas
        # Remove whitespace and newlines
        clean_content = re.sub(r'\s+', '', array_content)
        
        # Split by commas and convert to numbers (handle both ints and floats)
        try:
            numbers = []
            for x in clean_content.split(','):
                if x.strip():
                    # Try to parse as float first, then convert to int if it's a whole number
                    num = float(x.strip())
                    numbers.append(num)
            
            # Map level names to numbers
            level_map = {
                'AM': 1, 'BM': 2, 'CM': 3, 'DM': 4, 'EM': 5,
                'FM': 6, 'GM': 7, 'HM': 8, 'IM': 9, 'JM': 10
            }
            
            if level_name in level_map:
                level_num = level_map[level_name]
                levels[level_num] = {
                    'name': level_name,
                    'data': numbers
                }
                print(f"✅ Level {level_num} ({level_name}): {len(numbers)} integers")
            
        except ValueError as e:
            print(f"❌ Error parsing {level_name}: {e}")
    
    return levels

def main():
    js_file = "/Users/coldsauce/projects/spaceace/mapData.js"
    json_file = "/Users/coldsauce/projects/spaceace/map_data.json"
    
    print("🔄 Converting mapData.js to JSON...")
    
    try:
        levels = extract_map_data(js_file)
        
        if not levels:
            print("❌ No level data found!")
            return False
        
        # Write to JSON file
        with open(json_file, 'w') as f:
            json.dump(levels, f, indent=2)
        
        print(f"✅ Successfully converted {len(levels)} levels to {json_file}")
        print(f"   Available levels: {sorted(levels.keys())}")
        
        # Show some stats
        for level_num in sorted(levels.keys()):
            level = levels[level_num]
            data_len = len(level['data'])
            print(f"   Level {level_num} ({level['name']}): {data_len} integers")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)