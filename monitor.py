import subprocess
import time
import json
import shutil
import threading
import numpy as np
import csv
import os

# Try to import NVIDIA native bindings
try:
    import pynvml
except ImportError:
    pynvml = None

class HardwareMonitor:
    def __init__(self, platform):
        self.platform = platform
        self.has_rocm_smi = shutil.which("rocm-smi") is not None
        self.has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self.running = False
        self.history = {} 
        self.thread = None
        
        # Initialize NVML (NVIDIA Native) if available
        self.nvml_active = False
        if self.has_nvidia_smi and pynvml:
            try:
                pynvml.nvmlInit()
                self.nvml_active = True
            except Exception as e:
                print(f"[MONITOR] NVML Init failed: {e}. Falling back to nvidia-smi CLI.")
                self.nvml_active = False

    def get_gpu_count(self):
        if self.platform == "HIP" and self.has_rocm_smi:
            try:
                # AMD: Count keys in JSON output
                out = subprocess.check_output("rocm-smi -i --json", shell=True).decode()
                return len(json.loads(out))
            except: return 1
        elif self.has_nvidia_smi:
            if self.nvml_active:
                try:
                    return pynvml.nvmlDeviceGetCount()
                except: return 1
            else:
                try:
                    out = subprocess.check_output("nvidia-smi --query-gpu=count --format=csv,noheader", shell=True).decode()
                    return int(out.strip())
                except: return 1
        return 1

    def start_collection(self, gpu_ids, output_dir="."):
        self.running = True
        self.history = {gid: {'temp': [], 'pwr': [], 'clk': []} for gid in gpu_ids}
        
        # Initialize CSV Logging
        self.csv_file = open(os.path.join(output_dir, "time_series.csv"), "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Timestamp", "GPU_ID", "Temp_C", "Power_W", "Clock_MHz"])
        
        self.thread = threading.Thread(target=self._loop, args=(gpu_ids,))
        self.thread.start()

    def _loop(self, gpu_ids):
        start_t = time.time()
        while self.running:
            # Poll Data
            if self.platform == "HIP" and self.has_rocm_smi:
                self._poll_amd(gpu_ids)
            elif self.has_nvidia_smi:
                self._poll_nvidia(gpu_ids)
            
            # Log to CSV (Real-time)
            elapsed = round(time.time() - start_t, 2)
            for gid in gpu_ids:
                if gid in self.history and self.history[gid]['temp']:
                    # Get latest values
                    t = self.history[gid]['temp'][-1]
                    p = self.history[gid]['pwr'][-1] if self.history[gid]['pwr'] else 0
                    c = self.history[gid]['clk'][-1] if self.history[gid]['clk'] else 0
                    self.csv_writer.writerow([elapsed, gid, t, p, c])
            
            self.csv_file.flush()
            time.sleep(1)
        
        self.csv_file.close()
        
        # Shutdown NVML on exit if active
        if self.nvml_active:
            try: pynvml.nvmlShutdown()
            except: pass

    def stop_collection(self):
        self.running = False
        if self.thread: self.thread.join()
        return self._aggregate()

    def _poll_amd(self, gpu_ids):
        try:
            # -P (Power), -t (Temp), -c (Clock), -m (Mem Clock)
            out = subprocess.check_output("rocm-smi -P -t -c -m --json", shell=True).decode()
            data = json.loads(out)
            
            for gid in gpu_ids:
                # card0, card1... or gpu0... try to find the key
                card_key = f"card{gid}"
                keys = list(data.keys())
                
                # Robust Key Finding
                if card_key not in data: 
                    if len(keys) > gid: card_key = keys[gid]
                
                if card_key in data:
                    c = data[card_key]
                    
                    # 1. Temperature (Prioritize HBM/Junction, Fallback to Edge)
                    t_val = 0
                    for k, v in c.items():
                        k_lower = k.lower()
                        if "temp" in k_lower and "edge" in k_lower: 
                            t_val = float(v) # Baseline
                        if "temp" in k_lower and ("junction" in k_lower or "hotspot" in k_lower or "hbm" in k_lower):
                            t_val = float(v) # Upgrade to hotspot if found
                            break
                    if t_val > 0: self.history[gid]['temp'].append(t_val)

                    # 2. Power
                    p_val = 0
                    for k, v in c.items():
                        if "power" in k.lower() and "average" in k.lower():
                            p_val = float(v)
                            break
                        if "power" in k.lower() and float(v) > p_val:
                            p_val = float(v)
                    if p_val > 0: self.history[gid]['pwr'].append(p_val)

                    # 3. Clock (SCLK or MCLK)
                    clk_val = 0
                    for k, v in c.items():
                        if "sclk" in k.lower() and "(" in str(v):
                            # Format: "(1200Mhz)"
                            clean = str(v).replace("(", "").replace(")", "").replace("Mhz", "")
                            clk_val = float(clean)
                            break
                    if clk_val > 0: self.history[gid]['clk'].append(clk_val)

        except Exception as e:
            pass

    def _poll_nvidia(self, gpu_ids):
        # PATH A: Native NVML (Fast)
        if self.nvml_active:
            try:
                for gid in gpu_ids:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gid)
                    
                    # Temp
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    self.history[gid]['temp'].append(temp)
                    
                    # Power (mW -> W)
                    try:
                        pwr = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except: pwr = 0
                    self.history[gid]['pwr'].append(pwr)
                    
                    # Clock (MHz)
                    try:
                        clk = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    except: clk = 0
                    self.history[gid]['clk'].append(clk)
            except: 
                pass
        
        # PATH B: Legacy CLI Parsing (Slow Fallback)
        else:
            try:
                cmd = "nvidia-smi --query-gpu=index,temperature.gpu,power.draw,clocks.gr --format=csv,noheader,nounits"
                out = subprocess.check_output(cmd, shell=True).decode()
                for line in out.strip().split('\n'):
                    parts = line.split(',')
                    idx = int(parts[0])
                    if idx in self.history:
                        self.history[idx]['temp'].append(float(parts[1]))
                        self.history[idx]['pwr'].append(float(parts[2]))
                        self.history[idx]['clk'].append(float(parts[3]))
            except: pass

    def _aggregate(self):
        stats = {}
        for gid, data in self.history.items():
            # Calculate Averages/Max
            avg_temp = round(np.mean(data['temp']), 1) if data['temp'] else 0
            max_temp = round(np.max(data['temp']), 1) if data['temp'] else 0
            avg_pwr  = round(np.mean(data['pwr']), 1) if data['pwr'] else 0
            max_pwr  = round(np.max(data['pwr']), 1) if data['pwr'] else 0
            avg_clk  = round(np.mean(data['clk']), 0) if data['clk'] else 0
            max_clk  = round(np.max(data['clk']), 0) if data['clk'] else 0

            # --- Throttling Detection ---
            # Heuristic: If we hit high temps (>80C) and average clock is 
            # significantly lower (<85%) than max clock, we are likely throttling.
            throttled = False
            if max_temp > 80 and max_clk > 0:
                if avg_clk < (max_clk * 0.85):
                    throttled = True

            stats[gid] = {
                "avg_temp": avg_temp,
                "max_temp": max_temp,
                "avg_pwr":  avg_pwr,
                "max_pwr":  max_pwr,
                "avg_clk":  avg_clk,
                "max_clk":  max_clk,
                "throttled": throttled
            }
        return stats
