#ifndef MOCK_GPU_H
#define MOCK_GPU_H

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>

// --- MOCK TYPES ---
typedef int hipError_t;
#define hipSuccess 0
struct hipDeviceProp_t { int multiProcessorCount; };

// --- MOCK VECTOR TYPES ---
struct float4_ { float x, y, z, w; };
struct uint4_ { unsigned int x, y, z, w; };
struct float2_ { float x, y; };
typedef float4_ float4;
typedef uint4_ uint4;
typedef float2_ float2;

// --- MOCK QUALIFIERS ---
// Define these as empty strings so g++ ignores them
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(x)

// --- MOCK THREADING ---
// We simulate 1 thread per block for logic testing
struct uint3 { unsigned int x, y, z; };
static uint3 threadIdx = {0,0,0};
static uint3 blockIdx = {0,0,0};
static uint3 blockDim = {1,1,1};
static uint3 gridDim = {1,1,1};

inline void __syncthreads() {} // No-op for single thread

// --- MOCK API ---
inline hipError_t hipSetDevice(int dev) { return hipSuccess; }
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* p, int d) { p->multiProcessorCount = 1; return hipSuccess; }
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline hipError_t hipMemGetInfo(size_t* f, size_t* t) { *f=1e9; *t=2e9; return hipSuccess; }
inline hipError_t hipMalloc(void** p, size_t s) { *p = malloc(s); return hipSuccess; }
inline hipError_t hipFree(void* p) { free(p); return hipSuccess; }
inline hipError_t hipMemset(void* p, int v, size_t s) { memset(p, v, s); return hipSuccess; }

// --- MOCK INTRINSICS ---
// Half-precision shim (treat as float)
typedef float __half2;
inline float __float2half2_rn(float f) { return f; }
inline float __hfma2(float a, float b, float c) { return a*b+c; }
inline float __hneg2(float a) { return -a; }
inline float2 __half22float2(float a) { return {a, a}; }

// Constructors
inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return {x,y,z,w}; }

// Atomic Shim (Single threaded = standard add)
inline void atomicAdd(unsigned int* address, int val) { *address += val; }

#endif
