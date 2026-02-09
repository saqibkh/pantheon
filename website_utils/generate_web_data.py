import json
import os
import glob

# Paths
DB_DIR = "database"
OUTPUT_FILE = "docs/assets/web_data.json"

def main():
    results = []
    
    # Find all json reports
    files = glob.glob(os.path.join(DB_DIR, "pantheon_report_*.json"))
    
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                
                # Extract GPU Name (simplified)
                gpu_name = "Unknown GPU"
                if data.get("gpu_static_info"):
                    gpu_name = data["gpu_static_info"][0]["name"]
                
                # Process each test result in this file
                for test in data.get("test_results", []):
                    # Only keep valid runs with throughput
                    if test.get("Throughput (GB/s)") != "N/A":
                        results.append({
                            "gpu": gpu_name,
                            "test": test["Test Name"],
                            "throughput": float(test["Throughput (GB/s)"]),
                            "power": test["Avg Power (W)"],
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
