import argparse
import subprocess
import time
import os
import sys
import threading
import datetime
import json
import platform
import shutil
import pandas as pd
import numpy as np
import atexit
from monitor import HardwareMonitor

PANTHEON_VERSION = "1.0.0"

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Try to import psutil, warn if missing
try:
    import psutil
except ImportError:
    print("[PANTHEON] Warning: 'psutil' module not found. System info will be limited.")
    print("           Install it via: pip3 install psutil")
    psutil = None

# --- GLOBAL PROCESS TRACKER (For Cleanup) ---
ACTIVE_PROCS = []

def cleanup_zombies():
    """Kill any hanging subprocesses on exit."""
    global ACTIVE_PROCS
    for p in ACTIVE_PROCS:
        if p.poll() is None:  # If still running
            try:
                p.kill()
                print(f"[CLEANUP] Killed zombie process PID {p.pid}")
            except:
                pass

# Register immediately so it catches early crashes
atexit.register(cleanup_zombies)

# --- Configuration ---
BUILD_DIR = "build"
KERNEL_DIR = "kernels"
RESULTS_BASE_DIR = "results"
DATABASE_DIR = "database"

TEST_REGISTRY = {
    # Existing
    "hbm_write":      {"bin": "hbm_write",       "args": [], "desc": "HBM Write (Standard)"},
    "hbm_write_agg":  {"bin": "hbm_write_agg",   "args": [], "desc": "HBM Write (Aggressive)"},
    "hbm_read":       {"bin": "hbm_read",        "args": [], "desc": "HBM Read (Standard)"},
    "hbm_read_agg":   {"bin": "hbm_read_agg",    "args": [], "desc": "HBM Read (Aggressive)"},
    "voltage":        {"bin": "compute_virus",   "args": [], "desc": "Voltage Virus (ALU Hammer)"},
    "incinerator":    {"bin": "compute_virus_agg","args": [],"desc": "Incinerator (LDS Stress)"},
    "cache_lat":      {"bin": "cache_latency",   "args": [], "desc": "Cache Latency"},
    "sfu_stress":     {"bin": "sfu_stress",      "args": [], "desc": "SFU Virus (Transcendental Math)"},
    "pcie_bandwidth": {"bin": "pcie_bandwidth",  "args": [], "desc": "PCIe Thrasher (Host <-> Device)"},
    "pulse_virus":    {"bin": "pulse_virus",     "args": [], "desc": "Transient Pulse (VRM Attack 10Hz)"},
    "tensor_virus":   {"bin": "tensor_virus",    "args": [], "desc": "Tensor Virus (FP16 Matrix Power)"},
    "atomic_virus":   {"bin": "atomic_virus",    "args": [], "desc": "Atomic Virus (L2 Cache Thrash)"}
}

def log(msg):
    print(f"[PANTHEON] {msg}")

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

# --- System & GPU Info Gathering ---

def get_size(bytes, suffix="B"):
    """Scale bytes to proper format"""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def get_gpu_static_info():
    """Detects static GPU details (Name, VRAM, Driver) via CLI tools."""
    gpu_list = []
    
    # 1. Try NVIDIA
    if shutil.which("nvidia-smi"):
        try:
            cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"]
            out = subprocess.check_output(cmd, encoding="utf-8").strip()
            for line in out.split('\n'):
                idx, name, mem, driver = line.split(", ")
                gpu_list.append({
                    "id": int(idx),
                    "type": "NVIDIA",
                    "name": name,
                    "memory_total": f"{mem} MB",
                    "driver_version": driver
                })
        except: pass

    # 2. Try AMD
    if shutil.which("rocm-smi") and not gpu_list:
        try:
            # ROCm SMI logic is trickier to parse cleanly in one go, usually returns JSON
            out = subprocess.check_output("rocm-smi --showproductname --showmeminfo vram --json", shell=True, encoding="utf-8")
            data = json.loads(out)
            # data structure: {"card0": {"...": "..."}}
            for key, val in data.items():
                idx = int(key.replace("card", ""))
                # Extract Name
                name = val.get("Card Series", "Unknown AMD GPU")
                # Extract VRAM
                vram = val.get("VRAM Total Memory (B)", "0")
                if vram != "0":
                    vram = f"{int(vram) // (1024*1024)} MB"
                
                gpu_list.append({
                    "id": idx,
                    "type": "AMD",
                    "name": name,
                    "memory_total": vram,
                    "driver_version": "ROCm Driver" # Hard to get exact kernel module version via simple smi
                })
        except: pass

    return gpu_list

def get_toolkit_version(platform_name):
    """Get CUDA or ROCm/HIP version."""
    version = "Unknown"
    if platform_name == "CUDA":
        try:
            out = subprocess.check_output(["nvcc", "--version"], encoding="utf-8")
            for line in out.split('\n'):
                if "release" in line:
                    version = line.split("release ")[1].split(",")[0]
        except: pass
    elif platform_name == "HIP":
        try:
            # Try to read local version file common on ROCm installs
            if os.path.exists("/opt/rocm/.info/version"):
                with open("/opt/rocm/.info/version") as f:
                    version = f.read().strip()
            else:
                out = subprocess.check_output(["hipcc", "--version"], encoding="utf-8")
                version = "HIPCC Detected"
        except: pass
    return version

def get_system_snapshot(platform_name):
    """Aggregates all system info into a dictionary."""
    snapshot = {
        "pantheon_version": PANTHEON_VERSION,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "os_info": {
            "system": platform.system(),
            "release": platform.release(), # Kernel Version
            "version": platform.version(),
            "arch": platform.machine(),
        },
        "cpu_info": "psutil_missing",
        "ram_info": "psutil_missing",
        "gpu_static_info": get_gpu_static_info(),
        "toolkit_version": get_toolkit_version(platform_name)
    }

    if psutil:
        vm = psutil.virtual_memory()
        snapshot["cpu_info"] = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "freq": f"{psutil.cpu_freq().current:.2f}Mhz" if psutil.cpu_freq() else "N/A"
        }
        snapshot["ram_info"] = {
            "total": get_size(vm.total),
            "available": get_size(vm.available)
        }

    return snapshot

# --- Core Logic ---

def detect_platform():
    # 1. Force Mock Mode via Environment Variable (for CI)
    if os.environ.get("PANTHEON_MOCK") == "1":
        return "MOCK"
    
    # 2. Try Auto-Detect
    if shutil.which("hipcc"): return "HIP"
    if shutil.which("nvcc"): return "CUDA"
    
    # 3. Fallback to Mock if nothing else found (Optional, good for local dev without GPU)
    if shutil.which("g++"):
        print("[PANTHEON] Warning: No GPU compiler found. Defaulting to CPU Mock mode.")
        return "MOCK"
        
    return "UNKNOWN"

def build_kernels(platform):
    log(f"Detected Platform: {platform}. Compiling kernels...")
    ensure_dir(BUILD_DIR)
    cmd = ["make", f"PLATFORM={platform}"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Build Failed:\n" + result.stderr)
        sys.exit(1)
    log("Build Complete.")

def run_test(test_name, gpu_ids, duration, mem_pct):
    global ACTIVE_PROCS  # Allow modifying the global list
    config = TEST_REGISTRY[test_name]
    binary = os.path.join(BUILD_DIR, config["bin"])
    procs = []
    
    for gpu in gpu_ids:
        # Command: ./bin <gpu> <duration> <mem_pct> [args]
        cmd = [binary, str(gpu), str(duration), str(mem_pct)] + config["args"]
        log(f"Launching {test_name} on GPU {gpu} (Alloc: {mem_pct}% VRAM)...")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append((gpu, p))
        ACTIVE_PROCS.append(p)  # Track globaly

    return procs

def main():
    global ACTIVE_PROCS
    parser = argparse.ArgumentParser(description="PANTHEON: Universal GPU Stress Suite")
    parser.add_argument("--test", type=str, default="all", help=f"Test to run: {', '.join(TEST_REGISTRY.keys())} or 'all'")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds per test")
    parser.add_argument("--gpu", type=str, default="all", help="Comma separated list of GPU IDs (e.g. 0,1) or 'all'")
    parser.add_argument("--mem", type=int, default=99, help="Percentage of free VRAM to use (Default: 99)")
    args = parser.parse_args()

    # --- Setup ---
    platform = detect_platform()
    if platform == "UNKNOWN": sys.exit("Error: No compiler found.")
    
    build_kernels(platform)
    monitor = HardwareMonitor(platform)
    
    # --- GPU Discovery ---
    avail_count = monitor.get_gpu_count()
    if args.gpu == "all":
        target_gpus = list(range(avail_count))
    else:
        target_gpus = [int(x) for x in args.gpu.split(",")]

    # --- NEW: Display GPU Information ---
    print("\n" + "="*60)
    print("PANTHEON SYSTEM DETECTED (v{PANTHEON_VERSION})")
    print("="*60)
    gpu_info = get_gpu_static_info()
    if not gpu_info:
        print(f"Platform: {platform} (No detailed GPU info available via SMI)")
    else:
        for g in gpu_info:
            print(f"GPU {g['id']}: {g['name']} | {g['memory_total']} VRAM | Driver: {g['driver_version']}")
    print("="*60 + "\n")

    # --- Result Folder Setup ---
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RESULTS_BASE_DIR, timestamp_str)
    ensure_dir(run_dir)
    ensure_dir(DATABASE_DIR)
    
    log(f"Run ID: {timestamp_str}")

    # --- Test Selection ---
    if args.test == "all":
        queue = list(TEST_REGISTRY.keys())
    else:
        queue = [args.test]

    # --- Main Loop ---
    final_results = []

    for test in queue:
        log(f"--- STARTING TEST: {test.upper()} ---")
        
        # 1. Start Monitor
        monitor.start_collection(target_gpus, run_dir)

        # 2. Run Kernels
        # Reset active procs for this run (Global list accumulates, so we clear what's dead or manage it)
        ACTIVE_PROCS = [] 
        procs = run_test(test, target_gpus, args.duration, args.mem)
        
        # 3. Watchdog Wait Loop
        start_wait = time.time()
        timeout = args.duration + 10 
        
        try:
            while True:
                all_done = True
                for gpu, p in procs:
                    if p.poll() is None: # Still running
                        all_done = False
                
                if all_done:
                    break
                
                if time.time() - start_wait > timeout:
                    print(f"\n[WATCHDOG] Test exceeded {timeout}s limit. Terminating...")
                    for gpu, p in procs:
                        p.kill()
                    break
                
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[PANTHEON] Interrupted by user.")
            for gpu, p in procs: p.kill()
            sys.exit(0) # Exit immediately so atexit handles clean up if anything remains

        # 4. Stop Monitor & Get Stats
        hw_stats = monitor.stop_collection() 

        # 5. Parse Kernel Output
        for gpu, p in procs:
            out, err = p.communicate()
            throughput = "N/A"
            
            if p.returncode != 0:
                print(f"[ERROR] GPU {gpu} Test Failed (Code {p.returncode}):")
                if err: print(err.strip())
                if out: print(out.strip())

            if out:
                for line in out.split('\n'):
                    if "Throughput:" in line:
                        try:
                            raw_val = line.split(":")[1].strip()
                            number_part = raw_val.split(' ')[0] 
                            throughput = float(number_part)
                        except: 
                            pass
            
            stats = hw_stats.get(gpu, {})

            # Calculate Efficiency (MB/J)
            eff = 0
            if throughput != "N/A" and stats.get("avg_pwr", 0) > 0:
                eff = round((throughput * 1024) / stats.get("avg_pwr"), 2)

            row = {
                "Test Name": test,
                "Version": PANTHEON_VERSION,
                "Description": TEST_REGISTRY[test]["desc"],
                "GPU ID": gpu,
                "Duration (s)": args.duration,
                "Mem Usage (%)": args.mem,
                "Throughput (GB/s)": throughput,
                "Avg Temp (C)": stats.get("avg_temp", 0),
                "Max Temp (C)": stats.get("max_temp", 0),
                "Avg Power (W)": stats.get("avg_pwr", 0),
                "Max Power (W)": stats.get("max_pwr", 0),
                "Avg Clock (MHz)": stats.get("avg_clk", 0),
                
                # --- PRO METRICS ---
                "Efficiency (MB/J)": eff,
                "PCIe Gen": stats.get("pcie_gen", 0),
                "PCIe Width": stats.get("pcie_width", 0),
                "Limit Reason": stats.get("throttle_reason", "N/A"),
                "Max Mem Temp (C)": stats.get("max_mem_temp", 0),
                "Max Fan (%)": stats.get("max_fan", 0),
                "Volts Core (mV)": stats.get("max_volts_core", 0),
                "Volts SoC (mV)": stats.get("max_volts_soc", 0)
            }

            final_results.append(row)
            print(f"[RESULT] GPU {gpu} | {throughput} GB/s | {row['Max Temp (C)']}C Max | {row['Max Power (W)']}W Max")

        log(f"--- FINISHED TEST: {test.upper()} ---\n")

    # --- Report Generation ---
    df = pd.DataFrame(final_results)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)
    
    # CONSOLE: Drop the new noisy columns + Description
    cols_to_hide = ["Description", "Max Mem Temp (C)", "Max Fan (%)", 
                    "Volts Core (mV)", "Volts SoC (mV)", "Limit Reason", "Efficiency (MB/J)"]
    print(df.drop(columns=cols_to_hide, errors='ignore').to_string(index=False))
    
    print("="*80)

    # FILES: Save EVERYTHING (including new metrics)
    csv_path = os.path.join(run_dir, "summary.csv")
    xlsx_path = os.path.join(run_dir, "summary.xlsx")
    df.to_csv(csv_path, index=False)
    try: df.to_excel(xlsx_path, index=False)
    except: pass

    # 2. Save Full Database Snapshot (JSON) in database/ folder
    log("Generating Database Snapshot...")
    full_snapshot = get_system_snapshot(platform)
    full_snapshot["test_results"] = final_results
    
    db_file = os.path.join(DATABASE_DIR, f"pantheon_report_{timestamp_str}.json")
    
    with open(db_file, "w") as f:
        # Added cls=NumpyEncoder to automatically convert int64/float64 types
        json.dump(full_snapshot, f, indent=4, cls=NumpyEncoder)
    
    log(f"Snapshot saved to: {db_file}")


if __name__ == "__main__":
    main()
