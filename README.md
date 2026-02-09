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
