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
