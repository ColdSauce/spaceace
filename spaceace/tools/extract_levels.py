#!/usr/bin/env python3
"""
Extract SpaceAce level data from mapData.js and convert to JSON
"""

import re
import json

def extract_js_arrays(js_file_path):
    """Extract JavaScript const arrays from mapData.js"""
    with open(js_file_path, 'r') as f:
        content = f.read()
    
    # Find all const array declarations
    # Pattern matches: const VARNAME = [data]; (including $)
    pattern = r'const\s+([A-Z_$][A-Z0-9_$]*)\s*=\s*\[(.*?)\];'
    matches = re.findall(pattern, content, re.DOTALL)
    
    arrays = {}
    
    for var_name, array_content in matches:
        print(f"Processing {var_name}...")
        
        # Clean up the array content - remove whitespace and split by commas
        # Handle both integers and floats
        numbers = []
        for item in array_content.split(','):
            item = item.strip()
            if item and not item.startswith('//'):  # Skip empty items and comments
                try:
                    # Try to parse as number
                    if '.' in item:
                        numbers.append(float(item))
                    else:
                        numbers.append(int(item))
                except ValueError:
                    # Skip invalid numbers
                    continue
        
        arrays[var_name] = numbers
        print(f"  → {len(numbers)} numbers")
    
    return arrays

def create_level_mapping(arrays):
    """Create level number to array name mapping based on EXACT JavaScript logic"""
    # Map levels based on the exact JavaScript switch statement from index.html
    level_data = {}
    
    # Level 1: DL = AM;
    if "AM" in arrays:
        level_data["1"] = arrays["AM"]
        print(f"Level 1 → AM: {len(arrays['AM'])} elements")
    
    # Level 2: DL = BM;
    if "BM" in arrays:
        level_data["2"] = arrays["BM"]
        print(f"Level 2 → BM: {len(arrays['BM'])} elements")
    
    # Level 3: DL = CM.concat(DM, EM);
    if all(x in arrays for x in ["CM", "DM", "EM"]):
        combined = arrays["CM"] + arrays["DM"] + arrays["EM"]
        level_data["3"] = combined
        print(f"Level 3 → CM+DM+EM: {len(combined)} elements ({len(arrays['CM'])}+{len(arrays['DM'])}+{len(arrays['EM'])})")
    
    # Level 4: DL = FM;
    if "FM" in arrays:
        level_data["4"] = arrays["FM"]
        print(f"Level 4 → FM: {len(arrays['FM'])} elements")
    
    # Level 5: DL = GM;
    if "GM" in arrays:
        level_data["5"] = arrays["GM"]
        print(f"Level 5 → GM: {len(arrays['GM'])} elements")
    
    # Level 6: DL = HM;
    if "HM" in arrays:
        level_data["6"] = arrays["HM"]
        print(f"Level 6 → HM: {len(arrays['HM'])} elements")
    
    # Level 7: DL = IM;
    if "IM" in arrays:
        level_data["7"] = arrays["IM"]
        print(f"Level 7 → IM: {len(arrays['IM'])} elements")
    
    # Level 8: DL = JM;
    if "JM" in arrays:
        level_data["8"] = arrays["JM"]
        print(f"Level 8 → JM: {len(arrays['JM'])} elements")
    
    # Level 9: DL = KM;
    if "KM" in arrays:
        level_data["9"] = arrays["KM"]
        print(f"Level 9 → KM: {len(arrays['KM'])} elements")
    
    # Level 10: DL = _M.concat($M);
    if all(x in arrays for x in ["_M", "$M"]):
        combined = arrays["_M"] + arrays["$M"]
        level_data["10"] = combined
        print(f"Level 10 → _M+$M: {len(combined)} elements ({len(arrays['_M'])}+{len(arrays['$M'])})")
    
    # Also include the raw arrays for reference
    level_data["_raw_arrays"] = arrays
    level_data["_mapping_info"] = {
        "note": "Level mapping based on exact JavaScript switch statement from index.html",
        "level_3_combines": ["CM", "DM", "EM"],
        "level_10_combines": ["_M", "$M"]
    }
    
    return level_data

def main():
    print("🎯 Extracting SpaceAce level data from mapData.js")
    print("=" * 60)
    
    js_file = "mapData.js"
    json_file = "spaceace_levels.json"
    
    try:
        # Extract arrays from JavaScript
        arrays = extract_js_arrays(js_file)
        print(f"\n📊 Found {len(arrays)} arrays:")
        for name, data in arrays.items():
            print(f"  {name}: {len(data)} elements")
        
        # Create level mapping
        print(f"\n🗂️  Creating level mapping...")
        level_data = create_level_mapping(arrays)
        
        # Write to JSON
        print(f"\n💾 Writing to {json_file}...")
        with open(json_file, 'w') as f:
            json.dump(level_data, f, indent=2)
        
        print(f"✅ Successfully created {json_file}")
        
        # Verify the output
        print(f"\n🔍 Verification:")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        level_count = 0
        for key in data:
            if key.isdigit():
                level_num = int(key)
                level_count += 1
                print(f"  Level {level_num}: {len(data[key])} data points")
        
        print(f"\n✅ Total levels extracted: {level_count}")
        
    except FileNotFoundError:
        print(f"❌ Error: {js_file} not found")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()