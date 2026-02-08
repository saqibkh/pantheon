#include "../common/common.h"
#include <vector>
#include <chrono>

// Linear Congruential Generator constants for full-period traversal
#define LCG_A 1664525
#define LCG_C 1013904223

// --- INITIALIZATION KERNEL ---
__global__ void init_lcg_kernel(size_t* data, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // data[i] = (A*i + C) % n
    for (size_t i = tid; i < n; i += stride) {
        data[i] = (LCG_A * i + LCG_C) & (n - 1);
    }
}

// --- STRESS KERNEL ---
__global__ void latency_kernel(size_t* data, size_t n, int loops, size_t* sink) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        size_t idx = tid;
        // Pointer Chasing
        for (int i = 0; i < loops; ++i) {
            idx = data[idx];
        }
        sink[tid] = idx;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);

    CHECK(hipSetDevice(gpu_id));

    // 1. Calculate Main Buffer Size (The Walker)
    size_t free, total;
    CHECK(hipMemGetInfo(&free, &total));
    if (mem_pct > 99) mem_pct = 99;
    if (mem_pct < 1) mem_pct = 1;
    
    // Align to power of 2 for LCG
    size_t raw_size = (free * mem_pct) / 100;
    size_t num_elements = 1;
    while (num_elements * 2 * sizeof(size_t) <= raw_size) {
        num_elements *= 2;
    }
    size_t alloc_size = num_elements * sizeof(size_t);

    // 2. Allocate Main Buffer
    size_t* d_data; CHECK(hipMalloc(&d_data, alloc_size));

    // 3. Fix: Calculate Sink Size based on Thread Count (Not VRAM size)
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int num_blocks = prop.multiProcessorCount * 4;
    size_t sink_size = num_blocks * BLOCK_SIZE * sizeof(size_t);

    size_t* d_sink;
    CHECK(hipMalloc(&d_sink, sink_size));

    // 4. Initialize
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Init Random Walk on " 
              << alloc_size / (1024*1024) << " MB (" << num_elements << " nodes)..." << std::endl;
    LAUNCH_KERNEL(init_lcg_kernel, num_blocks, BLOCK_SIZE, d_data, num_elements);
    CHECK(hipDeviceSynchronize());    

    // 5. Run Stress
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running CACHE LATENCY STRESS..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int inner_loops = 10000;
    while (true) {
        LAUNCH_KERNEL(latency_kernel, num_blocks, BLOCK_SIZE, d_data, num_elements, inner_loops, d_sink);
        CHECK(hipDeviceSynchronize());
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    CHECK(hipFree(d_data));
    CHECK(hipFree(d_sink));
    return 0;
}
