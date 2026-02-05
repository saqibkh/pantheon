#ifndef PANTHEON_COMMON_H
#define PANTHEON_COMMON_H

#include "../common/hip_shim.h" 
#include <iostream>
#include <cstdlib>

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

// --- Non-Temporal Store (Bypass Cache & Write to HBM) ---
__device__ __forceinline__ void store_nt(void* addr, uint4 val) {
#ifdef __HIP_PLATFORM_AMD__
    typedef unsigned int __attribute__((vector_size(16))) vec_uint4;
    __builtin_nontemporal_store(*(vec_uint4*)&val, (vec_uint4*)addr);
#elif defined(__CUDACC__)
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" 
                 :: "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else
    *(uint4*)addr = val;
#endif
}

// --- Non-Temporal Load (Bypass Cache & Read from HBM) ---
__device__ __forceinline__ uint4 load_nt(void* addr) {
    uint4 ret;
#ifdef __HIP_PLATFORM_AMD__
    typedef unsigned int __attribute__((vector_size(16))) vec_uint4;
    vec_uint4 v = __builtin_nontemporal_load((vec_uint4*)addr);
    ret = *(uint4*)&v;
#elif defined(__CUDACC__)
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" 
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(addr));
#else
    ret = *(uint4*)addr;
#endif
    return ret;
}

#endif // PANTHEON_COMMON_H
