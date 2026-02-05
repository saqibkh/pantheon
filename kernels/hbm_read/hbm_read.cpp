#include "../common/common.h"
#include <vector>
#include <chrono>

// --- READ KERNEL ---
// Uses Non-Temporal Loads. Sums data to prevent optimization.
__global__ void hbm_read_kernel(uint4* data, size_t n, uint4* sink) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Accumulators (Break dependency chains)
    unsigned int acc = 0;

    for (size_t i = idx; i + stride * 3 < n; i += stride * 4) {
        uint4 v0 = load_nt(&data[i]);
        uint4 v1 = load_nt(&data[i+stride]);
        uint4 v2 = load_nt(&data[i+stride*2]);
        uint4 v3 = load_nt(&data[i+stride*3]);

        // Minimal logic to keep data alive
        acc += v0.x + v1.x + v2.x + v3.x;
    }

    // Write to sink (one per thread)
    if (acc == 0xDEADBEEF) sink[idx] = make_uint4(acc, 0, 0, 0);
}

// Helper to init memory so reads aren't checking zeros
__global__ void init_data(uint4* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    uint4 val = make_uint4(1, 2, 3, 4);
    for (size_t i = idx; i < n; i += stride) data[i] = val;
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

    uint4* d_data; CHECK(hipMalloc(&d_data, alloc_size));
    uint4* d_sink; CHECK(hipMalloc(&d_sink, alloc_size)); // Sink matches size to avoid bounds checks

    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int num_blocks = prop.multiProcessorCount * 20;

    // Init Data
    init_data<<<num_blocks, BLOCK_SIZE>>>(d_data, num_elements);
    CHECK(hipDeviceSynchronize());

    std::cout << "[PANTHEON] GPU " << gpu_id << ": HBM READ (NT Load) | " 
              << mem_pct << "% VRAM | " << duration << "s" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t bytes_transferred = 0;

    while (true) {
        hbm_read_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, num_elements, d_sink);
        CHECK(hipDeviceSynchronize());
        
        bytes_transferred += alloc_size; // Read only

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (bytes_transferred / 1e9) / seconds << " GB/s" << std::endl;

    CHECK(hipFree(d_data));
    CHECK(hipFree(d_sink));
    return 0;
}
