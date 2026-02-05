#include "../common/common.h"
#include <vector>
#include <chrono>

// --- STANDARD READ KERNEL (RDNA FIX) ---
// Uses volatile reads decomposed into 32-bit chunks to prevent crashes on AMD.
__global__ void hbm_read_kernel(uint4* data, size_t n, uint4* sink) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    unsigned int acc = 0;
    
    // Cast to volatile uint
    volatile unsigned int* ptr = (volatile unsigned int*)data;
    size_t num_ints = n * 4; 
    size_t step = stride * 4;

    // 4x Unroll (Standard)
    for (size_t i = idx * 4; i + step * 3 < num_ints; i += step * 4) {
        acc += ptr[i];
        acc += ptr[i + step];
        acc += ptr[i + step * 2];
        acc += ptr[i + step * 3];
    }

    if (acc == 0xDEADBEEF) sink[idx] = make_uint4(acc, 0, 0, 0);
}

// Helper to init data
__global__ void init_data(uint4* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    uint4 val = make_uint4(1, 1, 1, 1);
    for (size_t i = idx; i < n; i += stride) data[i] = val;
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);

    CHECK(hipSetDevice(gpu_id));

    size_t free, total; CHECK(hipMemGetInfo(&free, &total));
    if (mem_pct > 99) mem_pct = 99;
    
    // Safe Allocation Logic
    size_t alloc_size = (free * mem_pct) / 100;
    if (free - alloc_size < 50 * 1024 * 1024) alloc_size = free - 50 * 1024 * 1024;
    size_t num_elements = alloc_size / 16; 

    // Small Sink Allocation
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int num_blocks = prop.multiProcessorCount * 20;
    size_t sink_size = num_blocks * BLOCK_SIZE * sizeof(uint4);

    uint4* d_data; CHECK(hipMalloc(&d_data, alloc_size));
    uint4* d_sink; CHECK(hipMalloc(&d_sink, sink_size));

    init_data<<<num_blocks, BLOCK_SIZE>>>(d_data, num_elements);
    CHECK(hipDeviceSynchronize());

    std::cout << "[PANTHEON] GPU " << gpu_id << ": HBM READ (Standard) | " 
              << mem_pct << "% VRAM | " << duration << "s" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t bytes_transferred = 0;

    while (true) {
        hbm_read_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, num_elements, d_sink);
        CHECK(hipDeviceSynchronize());
        bytes_transferred += alloc_size;
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (bytes_transferred / 1e9) / seconds << " GB/s" << std::endl;

    CHECK(hipFree(d_data)); CHECK(hipFree(d_sink));
    return 0;
}
