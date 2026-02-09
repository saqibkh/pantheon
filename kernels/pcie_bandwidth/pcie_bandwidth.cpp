#include "../common/common.h"
#include <vector>
#include <chrono>

// --- PCIE BANDWIDTH THRASHER ---
// Stresses the interconnect by forcing massive bidirectional transfers.
// Detects: Bad Riser Cables, Unstable PCIe Generations (Gen4/5), Driver timeouts.

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    // mem_pct is used to size the transfer buffer (larger = better for peak bandwidth)

    CHECK(hipSetDevice(gpu_id));

    // 1. Allocate Pinned Host Memory (CPU side)
    // We use a large buffer (e.g., 256MB) to ensure we hit peak DMA speeds.
    size_t buffer_size = 256 * 1024 * 1024; 
    uint4* h_data;
    // hipHostMalloc allocates "pinned" (page-locked) memory for max PCIe speed
    CHECK(hipHostMalloc((void**)&h_data, buffer_size));

    // 2. Allocate Device Memory (GPU side)
    uint4* d_data;
    CHECK(hipMalloc(&d_data, buffer_size));

    // Initialize Host Data
    memset(h_data, 0xAA, buffer_size);

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running PCIE BANDWIDTH THRASHER (Host <-> Device)..." << std::endl;
    
    // Create Streams for Async Overlap (Read + Write simultaneously)
    hipStream_t stream1, stream2;
    CHECK(hipStreamCreate(&stream1));
    CHECK(hipStreamCreate(&stream2));

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t bytes_transferred = 0;

    // Warmup
    CHECK(hipMemcpyAsync(d_data, h_data, buffer_size, hipMemcpyHostToDevice, stream1));
    CHECK(hipDeviceSynchronize());

    while (true) {
        // Bidirectional Hammer:
        // Stream 1: Upload (H2D)
        // Stream 2: Download (D2H)
        // Note: In a real stress test, we just want to saturate the links. 
        // We ping-pong the data.
        
        CHECK(hipMemcpyAsync(d_data, h_data, buffer_size, hipMemcpyHostToDevice, stream1));
        CHECK(hipMemcpyAsync(h_data, d_data, buffer_size, hipMemcpyDeviceToHost, stream2));
        
        // Wait for both
        CHECK(hipStreamSynchronize(stream1));
        CHECK(hipStreamSynchronize(stream2));
        
        // Count: 2x buffer size per loop (Up + Down)
        bytes_transferred += (buffer_size * 2);

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (bytes_transferred / 1e9) / seconds << " GB/s" << std::endl;

    CHECK(hipStreamDestroy(stream1));
    CHECK(hipStreamDestroy(stream2));
    CHECK(hipFree(d_data));
    CHECK(hipHostFree(h_data));
    return 0;
}
