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
        # Reset history for this run
        self.history = {gid: {
            'temp_core': [], 'temp_mem': [], 
            'pwr': [], 'clk_core': [], 
            'fan_pct': [], 'volts_core': [], 'volts_soc': [],
            'pcie_gen': [], 'pcie_width': [], 'throttle': []
        } for gid in gpu_ids}
        
        self.csv_file = open(os.path.join(output_dir, "time_series.csv"), "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Timestamp", "GPU_ID", 
            "Temp_Core(C)", "Temp_Mem(C)", 
            "Power(W)", "Clock(MHz)", 
            "Fan(%)", "Volts_Core(mV)", "Volts_SoC(mV)",
            "PCIe_Gen", "PCIe_Width", "Throttle_Reason"
        ])
        
        self.thread = threading.Thread(target=self._loop, args=(gpu_ids,))
        self.thread.start()

    def _loop(self, gpu_ids):
        start_t = time.time()
        while self.running:
            try:
                if self.platform == "HIP" and self.has_rocm_smi:
                    self._poll_amd(gpu_ids)
                elif self.has_nvidia_smi:
                    self._poll_nvidia(gpu_ids)
            except Exception as e:
                # Prevent thread death if a single poll fails
                print(f"[MONITOR DEBUG] Polling error: {e}")
            
            elapsed = round(time.time() - start_t, 2)
            for gid in gpu_ids:
                h = self.history[gid]
                
                # Safe retrieval with defaults
                def get_last(key, default=0):
                    return h[key][-1] if h[key] else default

                t_c = get_last('temp_core')
                t_m = get_last('temp_mem')
                p   = get_last('pwr')
                c   = get_last('clk_core')
                f   = get_last('fan_pct')
                v_c = get_last('volts_core')
                v_s = get_last('volts_soc')
                pg  = get_last('pcie_gen')
                pw  = get_last('pcie_width')
                tr  = get_last('throttle', "N/A")
                    
                self.csv_writer.writerow([elapsed, gid, t_c, t_m, p, c, f, v_c, v_s, pg, pw, tr])
            
            self.csv_file.flush()
            time.sleep(1)
        
        self.csv_file.close()

    def stop_collection(self):
        self.running = False
        if self.thread: self.thread.join()
        return self._aggregate()

    # --- NVIDIA POLLING (Crash-Proof Version) ---
    def _poll_nvidia(self, gpu_ids):
        # Helper to safely parse floats/ints from "N/A" strings
        def safe_parse(val, type_func):
            try:
                val = val.strip()
                if val == "N/A" or val == "[Not Supported]": return 0
                if type_func == int and val.startswith("0x"): return int(val, 16)
                return type_func(val)
            except: return 0

        if self.nvml_active:
            # NVML Library Mode (Best)
            try:
                for gid in gpu_ids:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gid)
                    h = self.history[gid]
                    
                    try: h['temp_core'].append(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                    except: h['temp_core'].append(0)
                    
                    try: h['temp_mem'].append(pynvml.nvmlDeviceGetTemperature(handle, 2)) # 2 = Memory
                    except: h['temp_mem'].append(0)

                    try: h['pwr'].append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                    except: h['pwr'].append(0)
                    
                    try: h['clk_core'].append(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
                    except: h['clk_core'].append(0)

                    try: h['fan_pct'].append(pynvml.nvmlDeviceGetFanSpeed(handle))
                    except: h['fan_pct'].append(0)

                    # Voltage (Rarely supported on Linux Consumer)
                    h['volts_core'].append(0)
                    h['volts_soc'].append(0)

                    try: h['pcie_gen'].append(pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle))
                    except: h['pcie_gen'].append(0)

                    try: h['pcie_width'].append(pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))
                    except: h['pcie_width'].append(0)

                    # Throttle Reason
                    try:
                        reasons = []
                        mask = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                        if mask & pynvml.nvmlClocksThrottleReasonGpuIdle: reasons.append("Idle")
                        if mask & pynvml.nvmlClocksThrottleReasonSwPowerCap: reasons.append("Power")
                        if mask & pynvml.nvmlClocksThrottleReasonHwSlowdown: reasons.append("Thermal")
                        if not reasons: reasons.append("None")
                        h['throttle'].append("|".join(reasons))
                    except: h['throttle'].append("N/A")
            except: pass
        else:
            # CLI Fallback Mode (Robust)
            try:
                # Query: Index, Core Temp, Mem Temp, Power, Clock, Fan, Gen, Width, Throttle
                cmd = "nvidia-smi --query-gpu=index,temperature.gpu,temperature.memory,power.draw,clocks.gr,fan.speed,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active --format=csv,noheader,nounits"
                out = subprocess.check_output(cmd, shell=True).decode()
                
                for line in out.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) < 9: continue # Skip malformed lines

                    idx = safe_parse(parts[0], int)
                    if idx in self.history:
                        h = self.history[idx]
                        
                        h['temp_core'].append(safe_parse(parts[1], float))
                        h['temp_mem'].append(safe_parse(parts[2], float))
                        h['pwr'].append(safe_parse(parts[3], float))
                        h['clk_core'].append(safe_parse(parts[4], float))
                        h['fan_pct'].append(safe_parse(parts[5], float))
                        h['volts_core'].append(0)
                        h['volts_soc'].append(0)
                        h['pcie_gen'].append(safe_parse(parts[6], int))
                        h['pcie_width'].append(safe_parse(parts[7], int))
                        
                        # Throttle (Bitmask from CLI)
                        mask = safe_parse(parts[8], int)
                        reasons = []
                        if mask & 0x1: reasons.append("Idle")
                        if mask & 0x2: reasons.append("Power")
                        if mask & 0x8: reasons.append("Thermal")
                        h['throttle'].append("|".join(reasons) if reasons else "None")

            except Exception as e:
                # Print error only once to avoid spam, or ignore
                pass

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

                    # 1. Temperatures
                    t_core = 0; t_mem = 0
                    for k, v in c.items():
                        kl = k.lower()
                        if "temp" in kl:
                            try: val = float(v)
                            except: val = 0
                            if "edge" in kl: t_core = val
                            if "hbm" in kl or "mem" in kl or "junction" in kl: 
                                t_mem = max(t_mem, val)
                    h['temp_core'].append(t_core)
                    h['temp_mem'].append(t_mem)

                    # 2. Voltages
                    v_core = 0; v_soc = 0
                    for k, v in c.items():
                        kl = k.lower()
                        if "voltage" in kl:
                            try: val = float(v)
                            except: val = 0
                            if "0" in kl or "gfx" in kl: v_core = val
                            if "1" in kl or "soc" in kl: v_soc = val
                    h['volts_core'].append(v_core)
                    h['volts_soc'].append(v_soc)

                    # 3. Fan
                    f_pct = 0
                    for k, v in c.items():
                        if "fan" in k.lower() and "%" in str(k):
                            try: f_pct = float(v)
                            except: f_pct = 0
                    h['fan_pct'].append(f_pct)

                    # 4. Power/Clock
                    p_val = 0; clk_val = 0
                    for k, v in c.items():
                        if "power" in k.lower() and "average" in k.lower(): 
                            try: p_val = float(v)
                            except: p_val = 0
                        if "sclk" in k.lower() and "(" in str(v):
                            try:
                                clean = str(v).replace("(", "").replace(")", "").replace("Mhz", "")
                                clk_val = float(clean)
                            except: clk_val = 0
                    h['pwr'].append(p_val)
                    h['clk_core'].append(clk_val)

                    # 5. PCIe / Throttle (Placeholders for AMD JSON)
                    h['pcie_gen'].append(0)
                    h['pcie_width'].append(0)
                    h['throttle'].append("N/A")

        except Exception as e: pass

    def _aggregate(self):
        stats = {}
        for gid, data in self.history.items():
            def safe_max(l): return float(round(np.max(l), 1)) if l else 0
            def safe_mean(l): return float(round(np.mean(l), 1)) if l else 0
            def mode_str(l): return max(set(l), key=l.count) if l else "N/A"

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
                "pcie_gen": safe_max(data['pcie_gen']),
                "pcie_width": safe_max(data['pcie_width']),
                "throttle_reason": mode_str(data['throttle'])
            }
        return stats
