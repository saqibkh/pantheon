---
hide:
  - navigation
---

# PANTHEON

<div align="center">
  <img src="assets/logo.png" width="200" alt="Pantheon Logo">
  <h3>Universal GPU Stress & Diagnostics Suite</h3>
  <p>
    <a href="https://github.com/saqikhan/pantheon" class="md-button md-button--primary">
       View on GitHub
    </a>
    <a href="benchmarks/" class="md-button">
       Live Benchmarks
    </a>
  </p>
</div>

---

## Why Pantheon?

Pantheon is a cross-platform (CUDA/ROCm) stress testing tool designed to isolate and hammer specific GPU subsystems. Unlike generic benchmarks, Pantheon targets specific silicon limits.

<div class="grid cards" markdown>

-   **Memory Stress**
    ---
    Target the VRAM and Infinity Fabric with aggressive crosstalk patterns to detect bit flips and instability.
    
    *Test: `hbm_write_agg`*

-   **Tensor Cores**
    ---
    Saturate the matrix math pipelines with FP16 universal intrinsics to test maximum power draw.
    
    *Test: `tensor_virus`*

-   **VRM Transients**
    ---
    Oscillate load at 10Hz to induce high dI/dt transients, testing your power supply and voltage regulators.
    
    *Test: `pulse_virus`*

-   **L2 Cache & ROPs**
    ---
    Flood the memory subsystem with atomic operations to force constant cache locking and unlocking.
    
    *Test: `atomic_virus`*

</div>

## Quick Start

```bash
# Clone the repository
git clone [https://github.com/saqikhan/pantheon.git](https://github.com/saqikhan/pantheon.git)
cd pantheon

# Run the full suite (30 seconds per test)
python3 pantheon.py --test all --duration 30
```
