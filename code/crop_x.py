import json
import os
import glob

# 8am to 9am in seconds from midnight
RUSH_START = 28800  # 8 * 3600
RUSH_END   = 32400  # 9 * 3600

INPUT_DIR  = "./data/Xuancheng"
OUTPUT_DIR = "./data/xuancheng_mod"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all daily flow files
files = sorted(glob.glob(os.path.join(INPUT_DIR, "data_2023_*_type_filtered.json")))

if not files:
    print(f"No files found in {INPUT_DIR}")
    exit(1)

for fpath in files:
    fname = os.path.basename(fpath)
    print(f"\nProcessing {fname}...")

    with open(fpath, 'r') as f:
        flows = json.load(f)

    # Filter to rush hour window
    cropped = [v for v in flows if RUSH_START <= v['startTime'] <= RUSH_END]

    # Remap startTime to 0-based so simulation starts at t=0
    for v in cropped:
        v['startTime'] = v['startTime'] - RUSH_START
        v['endTime']   = v['endTime']   - RUSH_START

    # Output filename: keep date part only e.g. xuancheng_2023_04_03_8to9.json
    date_part = fname.replace("data_", "").replace("_type_filtered.json", "")
    out_name  = f"xuancheng_{date_part}_8to9.json"
    out_path  = os.path.join(OUTPUT_DIR, out_name)

    with open(out_path, 'w') as f:
        json.dump(cropped, f)

    print(f"  Original : {len(flows):>7,} vehicles")
    print(f"  Cropped  : {len(cropped):>7,} vehicles")
    print(f"  Saved to : {out_path}")

print("\nDone. All cropped files saved to", OUTPUT_DIR)