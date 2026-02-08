#ifndef PANTHEON_COMMON_H
#define PANTHEON_COMMON_H

#include <iostream>
#include <cstdlib>

// ==========================================
// PATH 1: CPU MOCK SIMULATION (CI/CD)
// ==========================================
#ifdef PANTHEON_MOCK
    #include "mock_gpu.h"
    
    // In Mock mode, "Launch" just calls the function directly on the CPU.
    // We ignore grid/block dimensions since we are single-threaded here.
    #define LAUNCH_KERNEL(kernel_name, grid, block, ...) kernel_name(__VA_ARGS__)

// ==========================================
// PATH 2: REAL GPU (CUDA / ROCm)
// ==========================================
#else
    // --- CROSS-PLATFORM SHIM (HIP -> CUDA) ---
    #ifdef __CUDACC__
        // NVIDIA / CUDA MODE
        #include <cuda_runtime.h>
        #include <device_launch_parameters.h>

        #define hipError_t cudaError_t
        #define hipSuccess cudaSuccess
        #define hipDeviceProp_t cudaDeviceProp
        #define hipGetErrorString cudaGetErrorString
        #define hipStream_t cudaStream_t
        #define hipSetDevice cudaSetDevice
        #define hipGetDeviceProperties cudaGetDeviceProperties
        #define hipMalloc cudaMalloc
        #define hipFree cudaFree
        #define hipMemcpy cudaMemcpy
        #define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
        #define hipMemGetInfo cudaMemGetInfo
        #define hipDeviceSynchronize cudaDeviceSynchronize
        #define hipStreamCreate cudaStreamCreate
        #define hipStreamDestroy cudaStreamDestroy
        #define hipStreamSynchronize cudaStreamSynchronize
        #define hipMemset cudaMemset 

    #else
        // AMD / ROCm MODE
        #include <hip/hip_runtime.h>
    #endif

    // Real Kernel Launch Syntax
    #define LAUNCH_KERNEL(kernel_name, grid, block, ...) kernel_name<<<grid, block>>>(__VA_ARGS__)

    // 128-bit Vector Types (GCC/Clang Vector Extensions for GPU)
    typedef float float4_ __attribute__((vector_size(16)));
    typedef unsigned int uint4_ __attribute__((vector_size(16)));
#endif


// ==========================================
// SHARED UTILITIES
// ==========================================

#define BLOCK_SIZE 256

#define CHECK(cmd) \
{ \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        std::cerr << "[PANTHEON ERROR] Code: " << error \
                  << " (" << hipGetErrorString(error) << ")" \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// --- NON-TEMPORAL STORE (Bypass Cache -> Write HBM) ---
// 1. CPU Mock: Standard Store
// 2. AMD: Builtin or Cast
// 3. CUDA: PTX ASM
__device__ __host__ __forceinline__ void store_nt(void* addr, uint4 val) {
#ifdef PANTHEON_MOCK
    // CPU Mock: Just write to memory
    *(uint4*)addr = val;
#elif defined(__HIP_PLATFORM_AMD__)
    // AMD: Use builtin for NT store
    typedef unsigned int __attribute__((vector_size(16))) vec_uint4;
    __builtin_nontemporal_store(*(vec_uint4*)&val, (vec_uint4*)addr);
#elif defined(__CUDACC__)
    // NVIDIA: PTX ASM
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" 
                 :: "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else
    *(uint4*)addr = val;
#endif
}

// --- NON-TEMPORAL LOAD (CRASH-PROOF VERSION) ---
// 1. CPU Mock: Standard Load
// 2. AMD: Decomposed Loads (RDNA Fix)
// 3. CUDA: PTX ASM
__device__ __host__ __forceinline__ uint4 load_nt(void* addr) {
    uint4 ret;
#ifdef PANTHEON_MOCK
    // CPU Mock: Just read memory
    ret = *(uint4*)addr;
#elif defined(__HIP_PLATFORM_AMD__)
    // RDNA CRASH FIX: Decompose into 4x 32-bit loads.
    // Attempting a single 128-bit vector load (*(uint4*)) segfaults on RDNA
    // if the buffer isn't perfectly 128-bit aligned.
    unsigned int* p = (unsigned int*)addr;
    ret.x = p[0];
    ret.y = p[1];
    ret.z = p[2];
    ret.w = p[3];
#elif defined(__CUDACC__)
    // NVIDIA: PTX Streaming Load
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" 
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(addr));
#else
    ret = *(uint4*)addr;
#endif
    return ret;
}

#endif // PANTHEON_COMMON_H
