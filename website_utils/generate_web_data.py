import json
import os
import glob

# Paths
DB_DIR = "database"
OUTPUT_FILE = "docs/assets/web_data.json"

def main():
    # Dictionary to store only the BEST result for each GPU+Test combo
    # Key: "GPU Name|Test Name" -> Value: Result Object
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
                
                # Extract GPU Name
                gpu_name = "Unknown GPU"
                if data.get("gpu_static_info") and len(data["gpu_static_info"]) > 0:
                    gpu_name = data["gpu_static_info"][0]["name"]
                
                # Process each test result in this file
                for test in data.get("test_results", []):
                    test_name = test["Test Name"]
                    
                    # --- SCORE NORMALIZATION ---
                    # If Throughput is "N/A" (e.g. Power Virus), we use Peak Power as the score.
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
                    # Unique Key = GPU + Test Name
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
    
    print(f"[Generate] Success. Reduced {len(files)} reports into {len(final_list)} unique best runs.")

if __name__ == "__main__":
    main()
