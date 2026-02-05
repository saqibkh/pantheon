#include "../common/common.h"
#include <vector>
#include <chrono>

// --- ATOMIC VIRUS ---
// Hammers memory with Atomic operations using a Wide Stride.
// This forces "Read-Modify-Write" cycles that overwhelm the L2 Cache Arbiters.
__global__ void atomic_virus_kernel(uint4* data, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Cast to uint for atomic operations
    unsigned int* atom_ptr = (unsigned int*)data;
    size_t num_ints = n * 4; 

    // Stride loop
    for (size_t i = tid; i < num_ints; i += stride) {
        // Atomic Add: Forces hardware to lock the cache line, update, and unlock.
        // Extremely taxing on the internal fabric.
        atomicAdd(&atom_ptr[i], 1);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);

    CHECK(hipSetDevice(gpu_id));

    size_t free, total; CHECK(hipMemGetInfo(&free, &total));
    if (mem_pct > 99) mem_pct = 99;
    
    // Use 60% VRAM. If too large, latency hides the stress. 
    // If too small, it stays in L1. 60% ensures L2 thrashing.
    size_t alloc_size = (free * mem_pct) / 100;
    size_t num_elements = alloc_size / 16; 

    void* d_data; CHECK(hipMalloc(&d_data, alloc_size));
    CHECK(hipMemset(d_data, 0, alloc_size)); 

    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int num_blocks = prop.multiProcessorCount * 16;

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running ATOMIC VIRUS (L2/ROP Stress)..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;

    while (true) {
        atomic_virus_kernel<<<num_blocks, BLOCK_SIZE>>>((uint4*)d_data, num_elements);
        CHECK(hipDeviceSynchronize());
        
        // 4 ints per uint4 element
        ops_performed += num_elements * 4;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    // Metric: MAPS (Million Atomic Operations Per Second)
    std::cout << "Throughput: " << (ops_performed / 1e6) / seconds << " MAPS" << std::endl;

    CHECK(hipFree(d_data));
    return 0;
}
