# Pantheon: Universal GPU Stress & Diagnostics Suite

Pantheon is a cross-platform (CUDA/ROCm) stress testing tool designed to isolate and hammer specific GPU subsystems. Unlike generic benchmarks (Furmark, 3DMark), Pantheon allows you to test specific silicon limits.

## Requirements
- **Python:** `pip3 install psutil pandas openpyxl numpy`
- **Compiler:** `nvcc` (NVIDIA) or `hipcc` (AMD)

## Quick Start
```bash
# Run the full suite (30 seconds per test)
python3 pantheon.py --test all --duration 30

# Run a specific "Power Virus" test
python3 pantheon.py --test tensor_virus --duration 60

# Run the full suite (30 seconds per test)
python3 pantheon.py --test all --duration 30

# Run a specific "VRM Cracker" test
python3 pantheon.py --test pulse_virus --duration 60
```

## Test Registry

| Test Name | Target Subsystem | Failure Symptoms |
| :--- | :--- | :--- |
| **`hbm_write_agg`** | VRAM & Infinity Fabric | Driver timeout, artifacts, system freeze. |
| **`hbm_read_agg`** | Memory Controller (IMC) | Stuttering, black screen. |
| **`tensor_virus`** | Tensor Cores (FP16 Matrix) | Maximum Power Draw, VRM shutdown (Black screen). |
| **`sfu_stress`** | Special Function Units (SIN/COS/DIV) | Arithmetic errors, high "Hotspot" temperatures. |
| **`pulse_virus`** | VRM Transients (dI/dt) | **Instant Shutdown** (Trips PSU/OCP protection). |
| **`pcie_bandwidth`** | PCIe Bus & DMA Engine | Low FPS, Stuttering, Audio crackling. |
| **`atomic_virus`** | L2 Cache & ROPs | Application crashes, erratic performance. |
| **`incinerator`** | Vector ALUs + SRAM | General instability, core clock drops. |
| **`cache_lat`** | Memory Latency Pointer Chasing | Random reboots, blue screens. |


## Interpretation of Results

The summary report (`results/<timestamp>/summary.xlsx`) contains detailed "Pro" metrics. Here is how to interpret them:

* **Efficiency (MB/J):** Calculated as `Throughput / Watts`.
    * **Healthy:** Stays relatively constant.
    * **Degraded:** If this drops significantly during a 1-hour burn-in, your silicon is "leaking" current (thermal runaway) or the VRMs are becoming inefficient due to heat.
* **PCIe Link:** Verifies the physical connection speed (e.g., `Gen4 x16`).
    * **Red Flag:** If it drops to `x8` or `Gen3` under load, check your riser cable, motherboard slot, or GPU mounting pressure.
* **Throttle Reason:** Tells you *why* performance is limited.
    * `[POWER]`: **Normal.** The card hit its TDP limit (expected for viruses).
    * `[THERMAL]`: **Critical.** The core is overheating (usually >83°C). Check thermal paste.
    * `[VOLTAGE]`: The VRMs cannot supply enough stable voltage.
* **Max Mem Temp:** The hottest point on your VRAM (HBM/GDDR6X).
    * **Note:** This is often 15-20°C hotter than the Core Temp. Keep this under 100°C to avoid permanent damage.
* **Throughput (GB/s):**
    * For `hbm_read` / `hbm_write`, this should be within 90% of your card's theoretical max bandwidth.
    * Low throughput = Memory Controller instability or aggressive error correction (ECC) kicking in.

## Website Dashboard

Pantheon includes a built-in web dashboard to visualize your benchmark results and compare different GPUs.

### 1. Install Dependencies
The dashboard is built with MkDocs. You need to install the material theme:
```bash
pip install mkdocs-material
```

### 2. Generate Data
The website reads from docs/assets/web_data.json. You must generate this file from your local database/ reports:
```bash
# parse local results and update the website JSON
python3 website_utils/generate_web_data.py
```

### 3. Run Local Server
Start the live preview server. It will auto-reload if you change any code or regenerate data.
```bash
mkdocs serve
```
Open https://www.google.com/search?q=http://127.0.0.1:8000 in your browser to view the performance leaderboard.
