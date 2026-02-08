#include "../common/common.h"
#include <chrono>

// --- VOLTAGE VIRUS KERNEL ---
// Uses volatile math to force rapid ALU state switching (0->1->0)
// This targets the Logic Rail (VDD_GFX) to induce di/dt droop.
__global__ void voltage_droop_kernel(int iters, float* sink) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Volatile prevents the compiler from pre-calculating the result
    volatile float a = 1.0f; 
    volatile float b = -1.0f;
    volatile float c = 0.5f;
    volatile float d = -0.5f;

    for(int i=0; i<iters; ++i) {
        // FMA (Fused Multiply Add) Chain
        // High power consumption density on both CDNA and Volta+ arch
        #ifdef __HIP_PLATFORM_AMD__
            a = __builtin_fma(a, b, c);
            b = __builtin_fma(b, c, d);
            c = __builtin_fma(c, d, a);
            d = __builtin_fma(d, a, b);
        #else
            // Standard C++ FMA (CUDA maps this to hardware FMA)
            a = fmaf(a, b, c);
            b = fmaf(b, c, d);
            c = fmaf(c, d, a);
            d = fmaf(d, a, b);
        #endif
        
        // Polarity Shock: Flip signs every 16 ops to maximize bit toggles
        if ((i & 0xF) == 0) {
            a = -a;
            b = -b;
        }
    }

    // Write to sink once at the end to keep the dependency chain alive
    if (a + b + c + d == 12345.0f) {
        sink[tid] = a;
    }
}

int main(int argc, char* argv[]) {
    // Expected args: ./bin <gpu> <duration> <mem_pct>
    if (argc < 4) return 1;
    
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    // int mem_pct = atoi(argv[3]); // Ignored: This test is Compute Bound

    CHECK(hipSetDevice(gpu_id));
    
    // Small allocation just for the sink (results)
    float* d_sink;
    CHECK(hipMalloc(&d_sink, 1024 * sizeof(float)));

    // Maximum Occupancy Strategy
    // We want every Compute Unit (CU) to be hammering the ALUs
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int num_blocks = prop.multiProcessorCount * 16; 

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running VOLTAGE DROOP VIRUS (ALU Hammer)..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Main Stress Loop
    while(true) {
        // Run a heavy batch of instructions
        LAUNCH_KERNEL(voltage_droop_kernel, num_blocks, BLOCK_SIZE, 20000, d_sink);
        CHECK(hipDeviceSynchronize());
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }
    
    CHECK(hipFree(d_sink));
    return 0;
}
