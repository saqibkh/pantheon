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
                out = subprocess.check_output("rocm-smi -i --json", shell=True).decode()
                return len(json.loads(out))
            except: return 1
        elif self.has_nvidia_smi:
            if self.nvml_active:
                try: return pynvml.nvmlDeviceGetCount()
                except: return 1
            else:
                try:
                    out = subprocess.check_output("nvidia-smi --query-gpu=count --format=csv,noheader", shell=True).decode()
                    return int(out.strip())
                except: return 1
        return 1

    def start_collection(self, gpu_ids, output_dir="."):
        self.running = True
        # Expanded Schema including PCIe and Throttle
        self.history = {gid: {
            'temp_core': [], 'temp_mem': [], 
            'pwr': [], 'clk_core': [], 
            'fan_pct': [], 'volts_core': [], 'volts_soc': [],
            'pcie_gen': [], 'pcie_width': [], 'throttle': [] # <--- NEW
        } for gid in gpu_ids}
        
        self.csv_file = open(os.path.join(output_dir, "time_series.csv"), "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Timestamp", "GPU_ID", 
            "Temp_Core(C)", "Temp_Mem(C)", 
            "Power(W)", "Clock(MHz)", 
            "Fan(%)", "Volts_Core(mV)", "Volts_SoC(mV)",
            "PCIe_Gen", "PCIe_Width", "Throttle_Reason" # <--- NEW
        ])
        
        self.thread = threading.Thread(target=self._loop, args=(gpu_ids,))
        self.thread.start()

    def _loop(self, gpu_ids):
        start_t = time.time()
        while self.running:
            if self.platform == "HIP" and self.has_rocm_smi:
                self._poll_amd(gpu_ids)
            elif self.has_nvidia_smi:
                self._poll_nvidia(gpu_ids)
            
            elapsed = round(time.time() - start_t, 2)
            for gid in gpu_ids:
                h = self.history[gid]
                if h['temp_core']:
                    # Safe retrieval with defaults
                    t_c = h['temp_core'][-1]
                    t_m = h['temp_mem'][-1] if h['temp_mem'] else 0
                    p   = h['pwr'][-1] if h['pwr'] else 0
                    c   = h['clk_core'][-1] if h['clk_core'] else 0
                    f   = h['fan_pct'][-1] if h['fan_pct'] else 0
                    v_c = h['volts_core'][-1] if h['volts_core'] else 0
                    v_s = h['volts_soc'][-1] if h['volts_soc'] else 0
                    
                    # --- FIX: Retrieve new metrics for CSV ---
                    pg  = h['pcie_gen'][-1] if h['pcie_gen'] else 0
                    pw  = h['pcie_width'][-1] if h['pcie_width'] else 0
                    tr  = h['throttle'][-1] if h['throttle'] else "N/A"
                    
                    # --- FIX: Write all 12 columns to match header ---
                    self.csv_writer.writerow([elapsed, gid, t_c, t_m, p, c, f, v_c, v_s, pg, pw, tr])
            
            self.csv_file.flush()
            time.sleep(1)
        
        self.csv_file.close()
        if self.nvml_active:
            try: pynvml.nvmlShutdown()
            except: pass

    def stop_collection(self):
        self.running = False
        if self.thread: self.thread.join()
        return self._aggregate()

    def _poll_amd(self, gpu_ids):
        try:
            # -v (Voltage), -P (Power), -t (Temp), -c (Clock), -f (Fan)
            out = subprocess.check_output("rocm-smi -v -P -t -c -f --json", shell=True).decode()
            data = json.loads(out)
            
            for gid in gpu_ids:
                card_key = f"card{gid}"
                keys = list(data.keys())
                if card_key not in data: 
                    if len(keys) > gid: card_key = keys[gid]
                
                if card_key in data:
                    c = data[card_key]
                    h = self.history[gid]

                    # 1. Temperatures (Core vs Mem)
                    t_core = 0; t_mem = 0
                    for k, v in c.items():
                        kl = k.lower()
                        if "temp" in kl:
                            val = float(v)
                            if "edge" in kl: t_core = val
                            # HBM / Junction / Mem
                            if "hbm" in kl or "mem" in kl or "junction" in kl: 
                                t_mem = max(t_mem, val) # Take hottest
                    h['temp_core'].append(t_core)
                    h['temp_mem'].append(t_mem)

                    # 2. Voltages (Core vs SoC)
                    # AMD usually labels them v0 (GFX) and v1 (SoC/Mem)
                    v_core = 0; v_soc = 0
                    for k, v in c.items():
                        kl = k.lower()
                        if "voltage" in kl:
                            val = float(v) # usually in mV
                            if "0" in kl or "gfx" in kl: v_core = val
                            if "1" in kl or "soc" in kl: v_soc = val
                    h['volts_core'].append(v_core)
                    h['volts_soc'].append(v_soc)

                    # 3. Fan
                    f_pct = 0
                    for k, v in c.items():
                        if "fan" in k.lower() and "%" in str(k):
                            f_pct = float(v)
                    h['fan_pct'].append(f_pct)

                    # 4. Power/Clock (Standard)
                    p_val = 0; clk_val = 0
                    for k, v in c.items():
                        if "power" in k.lower() and "average" in k.lower(): p_val = float(v)
                        if "sclk" in k.lower() and "(" in str(v):
                             clean = str(v).replace("(", "").replace(")", "").replace("Mhz", "")
                             clk_val = float(clean)
                    h['pwr'].append(p_val)
                    h['clk_core'].append(clk_val)

                    # 5. PCIe (AMD often reports in 'pcie_gen' or 'PCIE Generation')
                    # This varies by driver version, simple heuristic:
                    p_gen = 0; p_width = 0
                    if "pcie_gen" in c: p_gen = c["pcie_gen"]
                    if "pcie_lanes" in c: p_width = c["pcie_lanes"]
                    h['pcie_gen'].append(p_gen)
                    h['pcie_width'].append(p_width)

                    # 6. Throttle (AMD Throttling is complex via JSON, using placeholder)
                    h['throttle'].append("N/A")

        except Exception as e: pass

    def _poll_nvidia(self, gpu_ids):
        if self.nvml_active:
            try:
                for gid in gpu_ids:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gid)
                    h = self.history[gid]
                    
                    # Temp: Core
                    h['temp_core'].append(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                    
                    # Temp: Memory (If available, usually returns 0 or Error on older cards)
                    try:
                        # 2 = NVML_TEMPERATURE_MEM (Not exposed in older pynvml, hardcoding integer)
                        h['temp_mem'].append(pynvml.nvmlDeviceGetTemperature(handle, 2)) 
                    except: 
                        h['temp_mem'].append(0)

                    # Power
                    try: h['pwr'].append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                    except: h['pwr'].append(0)
                    
                    # Clock
                    try: h['clk_core'].append(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
                    except: h['clk_core'].append(0)

                    # Fan Speed
                    try: h['fan_pct'].append(pynvml.nvmlDeviceGetFanSpeed(handle))
                    except: h['fan_pct'].append(0)

                    # Voltage (NVIDIA rarely exposes this via NVML, usually 0)
                    h['volts_core'].append(0)
                    h['volts_soc'].append(0)

                    # PCIe
                    try: h['pcie_gen'].append(pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle))
                    except: h['pcie_gen'].append(0)

                    try: h['pcie_width'].append(pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))
                    except: h['pcie_width'].append(0)

                    # Throttle Reasons (Bitmask Decoding)
                    try:
                        reasons = []
                        mask = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                        if mask & pynvml.nvmlClocksThrottleReasonGpuIdle: reasons.append("Idle")
                        if mask & pynvml.nvmlClocksThrottleReasonSwPowerCap: reasons.append("Power")
                        if mask & pynvml.nvmlClocksThrottleReasonHwSlowdown: reasons.append("Thermal")
                        if mask & pynvml.nvmlClocksThrottleReasonSyncBoost: reasons.append("Sync")
                        if not reasons: reasons.append("None")
                        h['throttle'].append("|".join(reasons))
                    except:
                        h['throttle'].append("Unknown")
            except: pass
        else:
            # Fallback for CLI (NVIDIA) - Only Basic Metrics to prevent slow parsing
            try:
                cmd = "nvidia-smi --query-gpu=index,temperature.gpu,power.draw,clocks.gr --format=csv,noheader,nounits"
                out = subprocess.check_output(cmd, shell=True).decode()
                for line in out.strip().split('\n'):
                    parts = line.split(',')
                    idx = int(parts[0])
                    if idx in self.history:
                        self.history[idx]['temp_core'].append(float(parts[1]))
                        self.history[idx]['pwr'].append(float(parts[2]))
                        self.history[idx]['clk_core'].append(float(parts[3]))
            except: pass


    def _aggregate(self):
        stats = {}
        for gid, data in self.history.items():
            # Helper for safe max/mean
            def safe_max(l): return float(round(np.max(l), 1)) if l else 0
            def safe_mean(l): return float(round(np.mean(l), 1)) if l else 0

            avg_clk = safe_mean(data['clk_core'])
            max_clk = safe_max(data['clk_core'])
            max_temp = safe_max(data['temp_core'])

            # Throttling Heuristic
            throttled = False
            if max_temp > 80 and max_clk > 0:
                if avg_clk < (max_clk * 0.85):
                    throttled = True
            
            # Helper to find most common string
            def mode_str(l): return max(set(l), key=l.count) if l else "N/A"
            def safe_max(l): return round(np.max(l), 1) if l else 0

            stats[gid] = {
                "avg_temp": safe_mean(data['temp_core']),
                "max_temp": safe_max(data['temp_core']),
                "max_mem_temp": safe_max(data['temp_mem']),
                "avg_pwr":  safe_mean(data['pwr']),
                "max_pwr":  safe_max(data['pwr']),
                "avg_clk":  safe_mean(data['clk_core']),
                "max_fan": safe_max(data['fan_pct']),
                "max_volts_core": safe_max(data['volts_core']),
                "max_volts_soc": safe_max(data['volts_soc']),

                # New Aggregates
                "pcie_gen": safe_max(data['pcie_gen']),
                "pcie_width": safe_max(data['pcie_width']),
                "throttle_reason": mode_str(data['throttle'])
            }
        return stats
