import json
import os
import glob

# Paths
DB_DIR = "database"
OUTPUT_FILE = "docs/assets/web_data.json"

def main():
    # Dictionary to store the BEST result for each GPU+Test combo
    # Key: "GPU Name|Test Name" -> Value: Result Object
    best_runs = {}

    # 1. LOAD EXISTING DATA (Incremental Update)
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing_data = json.load(f)
                print(f"[Generate] Loaded {len(existing_data)} existing records from {OUTPUT_FILE}")
                
                for row in existing_data:
                    # Reconstruct the unique key
                    key = f"{row['gpu']}|{row['test']}"
                    best_runs[key] = row
        except Exception as e:
            print(f"[Generate] Warning: Could not read existing web_data.json: {e}")

    # 2. PROCESS NEW REPORTS
    files = glob.glob(os.path.join(DB_DIR, "pantheon_report_*.json"))
    if len(files) > 0:
        print(f"[Generate] Processing {len(files)} new report files from {DB_DIR}...")
    
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                
                # Create a map of ID -> Name for this specific report
                gpu_map = {}
                if data.get("gpu_static_info"):
                    for g in data["gpu_static_info"]:
                        gpu_map[g["id"]] = g["name"]
                
                # Process each test result
                for test in data.get("test_results", []):
                    test_name = test["Test Name"]
                    
                    # Resolve GPU Name
                    gid = test.get("GPU ID", 0)
                    gpu_name = gpu_map.get(gid, f"Unknown GPU {gid}")

                    # Score Normalization
                    raw_score = test.get("Throughput (GB/s)", "N/A")
                    score_val = 0.0
                    unit = "GB/s"

                    if raw_score != "N/A":
                        score_val = float(raw_score)
                    else:
                        score_val = float(test.get("Max Power (W)", 0))
                        unit = "Watts"

                    # Create Record
                    record = {
                        "gpu": gpu_name,
                        "test": test_name,
                        "version": data.get("pantheon_version", "Legacy"),
                        "score": score_val,
                        "unit": unit,
                        "temp_max": test.get("Max Temp (C)", 0),
                        "power_max": test.get("Max Power (W)", 0),
                        "clock_avg": test.get("Avg Clock (MHz)", 0),
                        "efficiency": test.get("Efficiency (MB/J)", 0),
                        "date": data["timestamp"]
                    }

                    # Unique Key
                    key = f"{gpu_name}|{test_name}"

                    # Merge Logic: Only overwrite if new score is BETTER
                    if key not in best_runs:
                        best_runs[key] = record
                    else:
                        # Compare new score vs existing score
                        if score_val > best_runs[key]["score"]:
                            best_runs[key] = record
                            
        except Exception as e:
            print(f"Skipping {f}: {e}")

    # 3. SAVE MERGED DATA
    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    final_list = list(best_runs.values())
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_list, f, indent=2)
    
    print(f"[Generate] Success. Database now contains {len(final_list)} unique best runs.")

if __name__ == "__main__":
    main()
