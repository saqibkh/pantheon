#include "../common/common.h"
#include <vector>
#include <chrono>

// --- SAFE READ KERNEL ---
// Uses volatile pointer access to prevent compiler optimization
// without using unstable intrinsics.
__global__ __launch_bounds__(256)
void hbm_read_agg_kernel(uint4* data, size_t n, uint4* sink) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    
    // Cast to volatile uint for safe, unoptimized reads
    volatile unsigned int* ptr = (volatile unsigned int*)data;
    size_t num_ints = n * 4; // uint4 = 4 uints
    size_t step = stride * 4; // Stride in uints

    // Manual 16x Unroll
    for (size_t i = idx * 4; i + step * 15 < num_ints; i += step * 16) {
        a0 += ptr[i];
        a1 += ptr[i + step];
        a2 += ptr[i + step * 2];
        a3 += ptr[i + step * 3];
        
        a0 += ptr[i + step * 4];
        a1 += ptr[i + step * 5];
        a2 += ptr[i + step * 6];
        a3 += ptr[i + step * 7];

        a0 += ptr[i + step * 8];
        a1 += ptr[i + step * 9];
        a2 += ptr[i + step * 10];
        a3 += ptr[i + step * 11];

        a0 += ptr[i + step * 12];
        a1 += ptr[i + step * 13];
        a2 += ptr[i + step * 14];
        a3 += ptr[i + step * 15];
    }

    // Write result to sink (One slot per thread)
    // idx is guaranteed to be < GridDim * BlockDim, so we map 1:1 to sink
    unsigned int final_sum = a0 + a1 + a2 + a3;
    if (final_sum == 0xDEADBEEF) sink[idx] = make_uint4(final_sum, 0, 0, 0);
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

    // 1. Calculate Available Memory
    size_t free, total; CHECK(hipMemGetInfo(&free, &total));
    if (mem_pct > 99) mem_pct = 99;
    
    // 2. Alloc Source Data (The massive buffer)
    // We leave 50MB padding to be safe against driver overhead
    size_t alloc_size = (free * mem_pct) / 100;
    if (free - alloc_size < 50 * 1024 * 1024) {
        alloc_size = free - 50 * 1024 * 1024;
    }
    size_t num_elements = alloc_size / 16; 

    // 3. Alloc Sink (Small buffer, just for results)
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int num_blocks = prop.multiProcessorCount * 20;
    size_t sink_size = num_blocks * BLOCK_SIZE * sizeof(uint4);

    uint4* d_data; CHECK(hipMalloc(&d_data, alloc_size));
    uint4* d_sink; CHECK(hipMalloc(&d_sink, sink_size)); // <-- FIX: Use small size

    // Init Data
    init_data<<<num_blocks, BLOCK_SIZE>>>(d_data, num_elements);
    CHECK(hipDeviceSynchronize());

    std::cout << "[PANTHEON] GPU " << gpu_id << ": HBM READ AGG | " 
              << mem_pct << "% VRAM | " << duration << "s" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t bytes_transferred = 0;

    while (true) {
        hbm_read_agg_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, num_elements, d_sink);
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
