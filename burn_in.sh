#!/bin/bash
# Save this as burn_in.sh and run: chmod +x burn_in.sh

echo "Starting Pantheon 1-Hour Burn-In..."

# Phase 1: Thermal Shock (10 Mins)
# Rapidly heats up the die to test cooling mount pressure
echo "[Phase 1] Thermal Shock (Tensor Virus)"
python3 pantheon.py --test tensor_virus --duration 600

# Phase 2: VRAM Saturation (10 Mins)
# Fills memory to 99% to test for bad memory chips
echo "[Phase 2] Memory Controller Saturation"
python3 pantheon.py --test hbm_read_agg --duration 600

# Phase 3: Mixed Load (40 Mins)
# Cycles through all tests to find transient instability
echo "[Phase 3] Mixed Cycle Loop"
for i in {1..3}
do
   echo "Loop $i / 3"
   python3 pantheon.py --test all --duration 300
done

echo "Burn-In Complete. Check database/ folder for logs."
