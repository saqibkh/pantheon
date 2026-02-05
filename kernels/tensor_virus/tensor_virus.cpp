#include "../common/common.h"
#include <chrono>

// --- HEADER FIX FOR CUDA/HIP ---
#ifdef __CUDACC__
    #include <cuda_fp16.h>
#else
    #include <hip/hip_fp16.h>
#endif

// --- TENSOR VIRUS (FP16 HAMMER) ---
// Uses Half-Precision (FP16) to saturate Tensor/Matrix cores.
__global__ void tensor_virus_kernel(int iters, float* sink) {
    size_t tid = threadIdx.x;
    
    // Initialize Half-Precision Registers
    // __half2 is a vector of two 16-bit floats (Packed FP16)
#ifdef __CUDACC__
    __half2 a = __float2half2_rn(1.0f);
    __half2 b = __float2half2_rn(0.5f);
    __half2 c = __float2half2_rn(-1.0f);
#else
    // AMD/ROCm often defines __half2 constructors differently in older versions,
    // but newer HIP supports the standard CUDA-style naming:
    __half2 a = __float2half2_rn(1.0f);
    __half2 b = __float2half2_rn(0.5f);
    __half2 c = __float2half2_rn(-1.0f);
#endif

    for(int i=0; i<iters; ++i) {
        // Universal FMA (Fused Multiply Add)
        // This works on both AMD (ROCm) and NVIDIA (CUDA)
        // It maps to v_pk_fma_f16 on AMD and HMMA/HFMA on NVIDIA.
        a = __hfma2(a, b, c);
        b = __hfma2(b, c, a);
        c = __hfma2(c, a, b);

        // Polarity flip to prevent value convergence
        if ((i & 0xF) == 0) {
            a = __hneg2(a);
        }
    }

    // Convert back to float to write sink
    float2 res = __half22float2(a);
    if (res.x + res.y == 12345.0f) sink[tid] = res.x;
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);

    CHECK(hipSetDevice(gpu_id));
    float* d_sink; CHECK(hipMalloc(&d_sink, 1024 * sizeof(float)));

    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    
    // Max Occupancy: Tensor tests need MASSIVE parallelism to fill the huge pipes.
    // For RDNA/gfx1010, we want to saturate the vector ALUs.
    int num_blocks = prop.multiProcessorCount * 32; 

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running TENSOR VIRUS (FP16 Stress)..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    while(true) {
        tensor_virus_kernel<<<num_blocks, BLOCK_SIZE>>>(20000, d_sink);
        CHECK(hipDeviceSynchronize());
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }
    
    CHECK(hipFree(d_sink));
    return 0;
}
