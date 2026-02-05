#include "../common/common.h"
#include <chrono>

// --- INCINERATOR KERNEL ---
// Combines FMA (Vector Unit) + LDS Thrashing (Local Data Share).
// This creates a "Power Virus" that attacks multiple sub-units simultaneously.

__global__ void incinerator_kernel(int iters, float* sink) {
    size_t tid = threadIdx.x;
    
    // 1. Setup Shared Memory (LDS)
    // We declare a volatile array to force high-speed SRAM access
    __shared__ volatile float smem[BLOCK_SIZE];
    
    // Init LDS
    smem[tid] = (float)tid;
    __syncthreads();

    // Registers for FMA
    volatile float a = 1.0f; 
    volatile float b = -1.0f;
    volatile float c = 0.5f;

    for(int i=0; i<iters; ++i) {
        // --- VECTOR UNIT STRESS (FMA) ---
        #ifdef __HIP_PLATFORM_AMD__
            a = __builtin_fma(a, b, c);
            b = __builtin_fma(b, c, a);
            c = __builtin_fma(c, a, b);
        #else
            a = fmaf(a, b, c);
            b = fmaf(b, c, a);
            c = fmaf(c, a, b);
        #endif
        
        // --- SRAM STRESS (LDS) ---
        // We read from shared memory, modify it, and write back.
        // We use an XOR index to cause bank conflicts (on purpose) for heat.
        int idx = tid ^ 1; // Swap neighbors
        float val = smem[idx];
        smem[tid] = val + 1.0f; // Write back
        
        // --- POLARITY SHOCK ---
        if ((i & 0xF) == 0) {
            a = -a;
        }
    }

    if (a + b + c + smem[tid] == 12345.0f) {
        sink[0] = a;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);

    CHECK(hipSetDevice(gpu_id));
    float* d_sink; CHECK(hipMalloc(&d_sink, 1024 * sizeof(float)));

    // Max Occupancy
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int num_blocks = prop.multiProcessorCount * 16; 

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running INCINERATOR (FMA + LDS Stress)..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    while(true) {
        incinerator_kernel<<<num_blocks, BLOCK_SIZE>>>(20000, d_sink);
        CHECK(hipDeviceSynchronize());
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }
    
    CHECK(hipFree(d_sink));
    return 0;
}
