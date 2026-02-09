import json
import os
import glob

# Paths
DB_DIR = "database"
OUTPUT_FILE = "docs/assets/web_data.json"

def main():
    # Dictionary to store only the BEST result for each GPU+Test combo
    best_runs = {}
    
    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Find all json reports
    files = glob.glob(os.path.join(DB_DIR, "pantheon_report_*.json"))
    print(f"[Generate] Found {len(files)} report files.")
    
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                
                # --- FIX: Create a Map of GPU ID -> GPU Name ---
                # Instead of grabbing just index [0], we map all detected GPUs.
                gpu_map = {}
                if data.get("gpu_static_info"):
                    for g in data["gpu_static_info"]:
                        gpu_map[g["id"]] = g["name"]
                
                # Process each test result in this file
                for test in data.get("test_results", []):
                    test_name = test["Test Name"]
                    
                    # --- FIX: Retrieve the Correct Name for this Result ---
                    # Default to ID 0 if missing, but try to use the specific ID
                    gid = test.get("GPU ID", 0)
                    gpu_name = gpu_map.get(gid, f"Unknown GPU {gid}")
                    
                    # Optional: If you want to distinguish identical cards (e.g. two 3090s),
                    # uncomment the line below to append the ID to the name:
                    # gpu_name = f"{gpu_name} #{gid}"

                    # --- SCORE NORMALIZATION ---
                    raw_score = test.get("Throughput (GB/s)", "N/A")
                    score_val = 0.0
                    unit = "GB/s"

                    if raw_score != "N/A":
                        score_val = float(raw_score)
                    else:
                        # Fallback for Compute/Power viruses
                        score_val = float(test.get("Max Power (W)", 0))
                        unit = "Watts"

                    # Create the Record Object
                    record = {
                        "gpu": gpu_name,
                        "test": test_name,
                        "score": score_val,
                        "unit": unit,
                        "temp_max": test.get("Max Temp (C)", 0),
                        "power_max": test.get("Max Power (W)", 0),
                        "clock_avg": test.get("Avg Clock (MHz)", 0),
                        "efficiency": test.get("Efficiency (MB/J)", 0),
                        "date": data["timestamp"]
                    }

                    # --- AGGREGATION LOGIC ---
                    # Unique Key = GPU Name + Test Name
                    # This ensures an RTX 3090 and RTX 4090 are stored separately.
                    key = f"{gpu_name}|{test_name}"

                    # If this key doesn't exist, OR if this new score is higher, save it.
                    if key not in best_runs:
                        best_runs[key] = record
                    else:
                        if score_val > best_runs[key]["score"]:
                            best_runs[key] = record

        except Exception as e:
            print(f"Skipping {f}: {e}")

    # Convert dictionary back to a list
    final_list = list(best_runs.values())

    # Save for web
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_list, f, indent=2)
    
    print(f"[Generate] Success. Processed {len(final_list)} unique results.")

if __name__ == "__main__":
    main()
