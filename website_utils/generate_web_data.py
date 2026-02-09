import json
import os
import glob

# Paths
DB_DIR = "database"
OUTPUT_FILE = "docs/assets/web_data.json"

def main():
    results = []
    
    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Find all json reports
    files = glob.glob(os.path.join(DB_DIR, "pantheon_report_*.json"))
    
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                
                # Extract GPU Name
                gpu_name = "Unknown GPU"
                if data.get("gpu_static_info") and len(data["gpu_static_info"]) > 0:
                    gpu_name = data["gpu_static_info"][0]["name"]
                
                # Process each test result
                for test in data.get("test_results", []):
                    # We want ALL results, even if throughput is N/A (e.g. Power Viruses)
                    
                    # Normalize Score: If Throughput is N/A, use Max Power
                    score = test.get("Throughput (GB/s)", "N/A")
                    score_unit = "GB/s"
                    
                    if score == "N/A":
                        score = test.get("Max Power (W)", 0)
                        score_unit = "Watts"

                    results.append({
                        "gpu": gpu_name,
                        "test": test["Test Name"],
                        "score": score,
                        "unit": score_unit,
                        "temp_max": test.get("Max Temp (C)", 0),
                        "power_max": test.get("Max Power (W)", 0),
                        "clock_avg": test.get("Avg Clock (MHz)", 0),
                        "efficiency": test.get("Efficiency (MB/J)", 0),
                        "date": data["timestamp"]
                    })
        except Exception as e:
            print(f"Skipping {f}: {e}")

    # Save for web
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Web data generated: {len(results)} records.")

if __name__ == "__main__":
    main()
