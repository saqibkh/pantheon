import json
import os
import glob
import numpy as np

# Paths
DB_DIR = "database"
OUTPUT_FILE = "docs/assets/web_data.json"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    best_runs = {}
    
    # 1. LOAD EXISTING
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing_data = json.load(f)
                for row in existing_data:
                    key = f"{row['gpu']}|{row['test']}"
                    best_runs[key] = row
        except: pass

    # 2. PROCESS NEW REPORTS
    files = glob.glob(os.path.join(DB_DIR, "pantheon_report_*.json"))
    
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                
                gpu_map = {}
                if data.get("gpu_static_info"):
                    for g in data["gpu_static_info"]:
                        gpu_map[g["id"]] = g["name"]
                
                for test in data.get("test_results", []):
                    test_name = test["Test Name"]
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

                    # --- CAPTURE ALL FIELDS ---
                    record = {
                        "gpu": gpu_name,
                        "test": test_name,
                        "version": data.get("pantheon_version", "1.0.0"),
                        "score": score_val,
                        "unit": unit,
                        # Standard Metrics
                        "temp_max": test.get("Max Temp (C)", 0),
                        "power_max": test.get("Max Power (W)", 0),
                        "clock_avg": test.get("Avg Clock (MHz)", 0),
                        "date": data["timestamp"],
                        # Pro Metrics (Previously Ignored)
                        "efficiency": test.get("Efficiency (MB/J)", 0),
                        "pcie_gen": test.get("PCIe Gen", 0),
                        "pcie_width": test.get("PCIe Width", 0),
                        "throttle": test.get("Limit Reason", "N/A"),
                        "temp_mem": test.get("Max Mem Temp (C)", 0),
                        "fan_max": test.get("Max Fan (%)", 0),
                        "volts_core": test.get("Volts Core (mV)", 0),
                        "volts_soc": test.get("Volts SoC (mV)", 0)
                    }

                    key = f"{gpu_name}|{test_name}"
                    if key not in best_runs:
                        best_runs[key] = record
                    else:
                        if score_val > best_runs[key]["score"]:
                            best_runs[key] = record
                            
        except Exception as e:
            print(f"Skipping {f}: {e}")

    # 3. SAVE
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(list(best_runs.values()), f, indent=2, cls=NumpyEncoder)
    
    print(f"[Generate] Database updated with {len(best_runs)} records.")

if __name__ == "__main__":
    main()
