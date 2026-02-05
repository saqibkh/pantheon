#include "../common/common.h"
#include <vector>
#include <chrono>

// --- WRITE KERNEL ---
// Uses Non-Temporal Stores to fill HBM bandwidth without polluting L2
__global__ void hbm_write_kernel(uint4* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Rail-to-Rail Pattern (0x00 <-> 0xFF)
    uint4 p0 = make_uint4(0x00000000, 0x00000000, 0x00000000, 0x00000000);
    uint4 p1 = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

    // Unroll 16x
    for (size_t i = idx; i + stride * 15 < n; i += stride * 16) {
        store_nt(&data[i], p0);
        store_nt(&data[i+stride], p1);
        store_nt(&data[i+stride*2], p0);
        store_nt(&data[i+stride*3], p1);
        store_nt(&data[i+stride*4], p0);
        store_nt(&data[i+stride*5], p1);
        store_nt(&data[i+stride*6], p0);
        store_nt(&data[i+stride*7], p1);
        store_nt(&data[i+stride*8], p0);
        store_nt(&data[i+stride*9], p1);
        store_nt(&data[i+stride*10], p0);
        store_nt(&data[i+stride*11], p1);
        store_nt(&data[i+stride*12], p0);
        store_nt(&data[i+stride*13], p1);
        store_nt(&data[i+stride*14], p0);
        store_nt(&data[i+stride*15], p1);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);

    CHECK(hipSetDevice(gpu_id));

    size_t free, total;
    CHECK(hipMemGetInfo(&free, &total));
    if (mem_pct > 99) mem_pct = 99;
    size_t alloc_size = (free * mem_pct) / 100;
    size_t num_elements = alloc_size / 16; 

    void* d_data;
    CHECK(hipMalloc(&d_data, alloc_size));

    // Oversubscribe CUs
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int num_blocks = prop.multiProcessorCount * 20;

    std::cout << "[PANTHEON] GPU " << gpu_id << ": HBM WRITE (NT Store) | " 
              << mem_pct << "% VRAM | " << duration << "s" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t bytes_transferred = 0;

    while (true) {
        hbm_write_kernel<<<num_blocks, BLOCK_SIZE>>>((uint4*)d_data, num_elements);
        CHECK(hipDeviceSynchronize());
        
        bytes_transferred += alloc_size; // Write only

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (bytes_transferred / 1e9) / seconds << " GB/s" << std::endl;

    CHECK(hipFree(d_data));
    return 0;
}
