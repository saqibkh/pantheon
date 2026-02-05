import subprocess
import time
import json
import shutil
import threading
import numpy as np

class HardwareMonitor:
    def __init__(self, platform):
        self.platform = platform
        self.has_rocm_smi = shutil.which("rocm-smi") is not None
        self.has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self.running = False
        self.history = {} # {gpu_id: {'temp': [], 'pwr': [], 'clk': []}}
        self.thread = None

    def get_gpu_count(self):
        if self.platform == "HIP" and self.has_rocm_smi:
            try:
                out = subprocess.check_output("rocm-smi -i --json", shell=True)
                return len(json.loads(out))
            except: return 1
        elif self.has_nvidia_smi:
            try:
                out = subprocess.check_output("nvidia-smi --query-gpu=count --format=csv,noheader", shell=True)
                return int(out.strip())
            except: return 1
        return 1

    def start_collection(self, gpu_ids):
        self.running = True
        self.history = {gid: {'temp': [], 'pwr': [], 'clk': []} for gid in gpu_ids}
        self.thread = threading.Thread(target=self._loop, args=(gpu_ids,))
        self.thread.start()

    def stop_collection(self):
        self.running = False
        if self.thread: self.thread.join()
        return self._aggregate()

    def _loop(self, gpu_ids):
        while self.running:
            if self.platform == "HIP" and self.has_rocm_smi:
                self._poll_amd(gpu_ids)
            elif self.has_nvidia_smi:
                self._poll_nvidia(gpu_ids)
            time.sleep(1)

    def _poll_amd(self, gpu_ids):
        try:
            out = subprocess.check_output("rocm-smi -P -t -c --json", shell=True)
            data = json.loads(out)
            for gid in gpu_ids:
                key = f"card{gid}"
                if key in data:
                    c = data[key]
                    try:
                        self.history[gid]['temp'].append(float(c.get("Temperature (Sensor edge)", 0)))
                        self.history[gid]['pwr'].append(float(c.get("Average Power", 0)))
                        # Clock string might be like "(1200Mhz)"
                        clk_raw = c.get("sclk clock speed:", "(0Mhz)")
                        clk = float(clk_raw.replace("(", "").replace(")", "").replace("Mhz", ""))
                        self.history[gid]['clk'].append(clk)
                    except: pass
        except: pass

    def _poll_nvidia(self, gpu_ids):
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
            stats[gid] = {
                "avg_temp": round(np.mean(data['temp']), 1) if data['temp'] else 0,
                "max_temp": round(np.max(data['temp']), 1) if data['temp'] else 0,
                "avg_pwr":  round(np.mean(data['pwr']), 1) if data['pwr'] else 0,
                "max_pwr":  round(np.max(data['pwr']), 1) if data['pwr'] else 0,
                "avg_clk":  round(np.mean(data['clk']), 0) if data['clk'] else 0,
            }
        return stats
