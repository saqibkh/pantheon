#ifndef HIP_SHIM_H
#define HIP_SHIM_H

// If compiling with NVCC (NVIDIA), map HIP -> CUDA
#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
    #include <iostream>

    // 1. Rename Types
    #define hipError_t cudaError_t
    #define hipSuccess cudaSuccess
    #define hipDeviceProp_t cudaDeviceProp
    #define hipGetErrorString cudaGetErrorString
    #define hipStream_t cudaStream_t

    // 2. Rename Functions
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

    // 3. Handle Kernel Launch Syntax if needed (usually compatible)
    
#else
    // If compiling with HIPCC (AMD), just use standard HIP
    #include <hip/hip_runtime.h>
#endif

#endif // HIP_SHIM_H
