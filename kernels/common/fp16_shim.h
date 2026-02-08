#ifndef PANTHEON_FP16_SHIM_H
#define PANTHEON_FP16_SHIM_H

#ifdef __CUDACC__
    #include <cuda_fp16.h>
    // Map CUDA intrinsics to generic names if needed, or use them directly
#else
    #include <hip/hip_fp16.h>
#endif

// Universal FP16 Constructors & Math
// This hides the architecture differences for __half2 initialization
__device__ __forceinline__ __half2 make_half2_universal(float f) {
#ifdef __CUDACC__
    return __float2half2_rn(f);
#else
    // Newer ROCm/HIP supports this, older versions might need a cast
    return __float2half2_rn(f); 
#endif
}

#endif // PANTHEON_FP16_SHIM_H
