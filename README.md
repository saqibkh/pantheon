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

Test RegistryTest NameTarget SubsystemFailure Symptomshbm_readVRAM Controller (Read)Screen artifacts, system freeze.hbm_write_aggInfinity Fabric / L2 CacheDriver timeout, massive stutter.tensor_virusTensor/Matrix Cores (FP16)Thermal Throttling, VRM shutdown (Black screen).atomic_virusROPs & L2 AtomicsApplication crashes, erratic performance.incineratorVector ALUs + SRAMGeneral instability, core clock drops.Interpretation of ResultsThroughput: For Read tests, this should match your card's theoretical max bandwidth.Power: tensor_virus should generate the highest power draw (W).Temps: incinerator or tensor_virus should generate the highest Hotspot temps.
### **Bonus: The "Burn-In" Script**
If you want to use this to test the stability of an overclock or a used GPU, create this shell script to run a 1-hour cyclic burn-in.

**Create File: `burn_in.sh`**

```bash
#!/bin/bash
echo "Starting 1-Hour Stability Burn-In..."

# 1. Heat up the loop (Matrix Cores) - 10 Minutes
echo "[Phase 1] Thermal Shock (Tensor Virus)"
python3 pantheon.py --test tensor_virus --duration 600

# 2. Thrash the Memory Controller - 10 Minutes
echo "[Phase 2] Memory Controller Saturation"
python3 pantheon.py --test hbm_read_agg --duration 600

# 3. Thrash the Cache/Fabric - 10 Minutes
echo "[Phase 3] Fabric/L2 Stress"
python3 pantheon.py --test atomic_virus --duration 600

# 4. Mixed Loop (All tests cycling) - 30 Minutes
echo "[Phase 4] Mixed Cycle"
for i in {1..5}
do
   python3 pantheon.py --test all --duration 60
done

echo "Burn-In Complete. Check database/ folder for logs."
You can make it executable with chmod +x burn_in.sh
