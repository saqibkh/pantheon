#include "../common/common.h"
#include <chrono>

// --- SFU (SPECIAL FUNCTION UNIT) VIRUS ---
// Hammers the transcendental math pipelines (SIN, COS, EXP, LOG, RSQRT).
// These units often share power rails with Texture units or Tensor cores.

__global__ void sfu_stress_kernel(int iters, float* sink) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Seed with thread ID to prevent caching
    float a = (float)tid * 0.0001f;
    float b = 1.0f;

    for(int i=0; i<iters; ++i) {
        // The "Transcendental Torture" Chain
        // High-latency, low-throughput instructions
        a = sinf(a) * cosf(b);
        b = expf(a) / (1.0f + fabsf(a));
        a = logf(fabsf(b) + 0.00001f);
        b = rsqrtf(a * a + 1.0f);
        
        // Periodic perturbation to avoid convergence to 0/INF
        if ((i & 0x1F) == 0) {
            a += 0.1f;
            b = 1.0f - b;
        }
    }

    // Write dependency to sink
    if (a > 100000.0f) sink[tid] = b; // Conditional write (rarely taken, prevents optimizing out)
    sink[tid] = a;
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);

    CHECK(hipSetDevice(gpu_id));
    float* d_sink; CHECK(hipMalloc(&d_sink, 1024 * sizeof(float)));

    // Max Occupancy Strategy
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int num_blocks = prop.multiProcessorCount * 32; 

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running SFU VIRUS (Transcendental Math)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while(true) {
        LAUNCH_KERNEL(sfu_stress_kernel, num_blocks, BLOCK_SIZE, 5000, d_sink);
        CHECK(hipDeviceSynchronize());
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }
    
    CHECK(hipFree(d_sink));
    return 0;
}
