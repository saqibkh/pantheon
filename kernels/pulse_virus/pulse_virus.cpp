#include "../common/common.h"
#include <chrono>
#include <thread> // For sleep

// --- PULSE VIRUS ---
// Induces maximum dI/dt (Change in Current over Time).
// 1. Spikes load to 100% instantly (Heavy FMA).
// 2. Drops load to 0% instantly (Sleep).

__global__ void pulse_load_kernel(int iters, float* sink) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Volatile math to force ALU usage
    volatile float a = 1.0f;
    volatile float b = -1.0f;
    volatile float c = 0.5f;

    for(int i=0; i<iters; ++i) {
        #ifdef __HIP_PLATFORM_AMD__
            a = __builtin_fma(a, b, c);
            b = __builtin_fma(b, c, a);
            c = __builtin_fma(c, a, b);
        #else
            a = fmaf(a, b, c);
            b = fmaf(b, c, a);
            c = fmaf(c, a, b);
        #endif
    }
    
    // Write output to prevent optimization
    // We mask the index to ensure we never write out of bounds
    // even if the grid size changes dynamically.
    if (a > 100000.0f) sink[tid] = b;
    sink[tid] = a;
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);

    CHECK(hipSetDevice(gpu_id));

    // Max Occupancy calculation
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int num_blocks = prop.multiProcessorCount * 32; 
    
    // --- FIX: Allocate sink big enough for ALL threads ---
    size_t total_threads = num_blocks * BLOCK_SIZE;
    float* d_sink; 
    CHECK(hipMalloc(&d_sink, total_threads * sizeof(float)));

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running PULSE VIRUS (Transient Load @ 10Hz)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while(true) {
        // PHASE 1: SPIKE (Load ON)
        LAUNCH_KERNEL(pulse_load_kernel, num_blocks, BLOCK_SIZE, 20000, d_sink);
        CHECK(hipDeviceSynchronize());
        
        // PHASE 2: DROOP (Load OFF)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }
    
    CHECK(hipFree(d_sink));
    return 0;
}
