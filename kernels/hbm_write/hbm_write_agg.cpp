#include "../common/common.h"
#include <vector>
#include <chrono>

// --- AGGRESSIVE WRITE KERNEL ---
// Changes from Standard:
// 1. Crosstalk Pattern (0x55... vs 0xAA...)
// 2. 64x Unroll Factor (Max Bus Saturation)
// 3. Strict Launch Bounds

__global__ __launch_bounds__(256)
void hbm_write_agg_kernel(uint4* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Pattern A: 10101010... (0xAAAAAAAA)
    // Pattern B: 01010101... (0x55555555)
    // Switching between these flips EVERY bit on the bus relative to its neighbor.
    // This creates maximum Inter-Symbol Interference (ISI) and Crosstalk.
    uint4 pA = make_uint4(0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA);
    uint4 pB = make_uint4(0x55555555, 0x55555555, 0x55555555, 0x55555555);

    // Massive 64x Unroll
    // We want the instruction pipeline to be nothing but "STORE, STORE, STORE"
    for (size_t i = idx; i + stride * 63 < n; i += stride * 64) {
        store_nt(&data[i],    pA); store_nt(&data[i+1],  pB);
        store_nt(&data[i+2],  pA); store_nt(&data[i+3],  pB);
        store_nt(&data[i+4],  pA); store_nt(&data[i+5],  pB);
        store_nt(&data[i+6],  pA); store_nt(&data[i+7],  pB);
        store_nt(&data[i+8],  pA); store_nt(&data[i+9],  pB);
        store_nt(&data[i+10], pA); store_nt(&data[i+11], pB);
        store_nt(&data[i+12], pA); store_nt(&data[i+13], pB);
        store_nt(&data[i+14], pA); store_nt(&data[i+15], pB);
        
        // Block 2
        store_nt(&data[i+16], pA); store_nt(&data[i+17], pB);
        store_nt(&data[i+18], pA); store_nt(&data[i+19], pB);
        store_nt(&data[i+20], pA); store_nt(&data[i+21], pB);
        store_nt(&data[i+22], pA); store_nt(&data[i+23], pB);
        store_nt(&data[i+24], pA); store_nt(&data[i+25], pB);
        store_nt(&data[i+26], pA); store_nt(&data[i+27], pB);
        store_nt(&data[i+28], pA); store_nt(&data[i+29], pB);
        store_nt(&data[i+30], pA); store_nt(&data[i+31], pB);

        // Block 3
        store_nt(&data[i+32], pA); store_nt(&data[i+33], pB);
        store_nt(&data[i+34], pA); store_nt(&data[i+35], pB);
        store_nt(&data[i+36], pA); store_nt(&data[i+37], pB);
        store_nt(&data[i+38], pA); store_nt(&data[i+39], pB);
        store_nt(&data[i+40], pA); store_nt(&data[i+41], pB);
        store_nt(&data[i+42], pA); store_nt(&data[i+43], pB);
        store_nt(&data[i+44], pA); store_nt(&data[i+45], pB);
        store_nt(&data[i+46], pA); store_nt(&data[i+47], pB);

        // Block 4
        store_nt(&data[i+48], pA); store_nt(&data[i+49], pB);
        store_nt(&data[i+50], pA); store_nt(&data[i+51], pB);
        store_nt(&data[i+52], pA); store_nt(&data[i+53], pB);
        store_nt(&data[i+54], pA); store_nt(&data[i+55], pB);
        store_nt(&data[i+56], pA); store_nt(&data[i+57], pB);
        store_nt(&data[i+58], pA); store_nt(&data[i+59], pB);
        store_nt(&data[i+60], pA); store_nt(&data[i+61], pB);
        store_nt(&data[i+62], pA); store_nt(&data[i+63], pB);

        // Standard pointer arithmetic handled by loop
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

    // --- FIX: Change void* to uint4* ---
    uint4* d_data;
    CHECK(hipMalloc(&d_data, alloc_size));

    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int num_blocks = prop.multiProcessorCount * 20;

    std::cout << "[PANTHEON] GPU " << gpu_id << ": HBM AGGRESSIVE WRITE (Crosstalk) | "
              << mem_pct << "% VRAM | " << duration << "s" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t bytes_transferred = 0;

    while (true) {
        // Use LAUNCH_KERNEL for CI compatibility
        LAUNCH_KERNEL(hbm_write_agg_kernel, num_blocks, BLOCK_SIZE, d_data, num_elements);
        CHECK(hipDeviceSynchronize());

        bytes_transferred += alloc_size;
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (bytes_transferred / 1e9) / seconds << " GB/s" << std::endl;

    CHECK(hipFree(d_data));
    return 0;
}
