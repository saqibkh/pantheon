#ifndef MOCK_GPU_H
#define MOCK_GPU_H

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <map>

// --- MOCK STATE ---
static std::map<void*, size_t> g_mock_allocations;
static size_t g_mock_total_bytes = 0;

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
#define __global__
#define __device__
#define __host__
#define __shared__
#define __forceinline__ inline
#define __launch_bounds__(x)

// --- MOCK THREADING ---
struct uint3 { unsigned int x, y, z; };
static uint3 threadIdx = {0,0,0};
static uint3 blockIdx = {0,0,0};
static uint3 blockDim = {1,1,1};
static uint3 gridDim = {1,1,1};

inline void __syncthreads() {} 

// --- MOCK API ---
inline hipError_t hipSetDevice(int dev) { return hipSuccess; }
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* p, int d) { p->multiProcessorCount = 1; return hipSuccess; }
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline hipError_t hipMemGetInfo(size_t* f, size_t* t) { *f=1e9; *t=2e9; return hipSuccess; }
inline const char* hipGetErrorString(hipError_t error) { return "Mock Success"; }

// --- TRACKED MEMORY MANAGEMENT ---

// Device Malloc
template <typename T>
inline hipError_t hipMalloc(T** p, size_t s) {
    *p = (T*)malloc(s);
    if (*p) {
        g_mock_allocations[*p] = s;
        g_mock_total_bytes += s;
    }
    return hipSuccess;
}

// Host Malloc (Pinned) - Simulates standard malloc for CPU
inline hipError_t hipHostMalloc(void** p, size_t s) {
    *p = malloc(s);
    if (*p) {
        g_mock_allocations[*p] = s;
        g_mock_total_bytes += s;
    }
    return hipSuccess;
}

// Universal Free (Handles both Device and Host pointers in Mock)
inline hipError_t hipFree(void* p) {
    if (p) {
        if (g_mock_allocations.find(p) != g_mock_allocations.end()) {
            g_mock_total_bytes -= g_mock_allocations[p];
            g_mock_allocations.erase(p);
        }
        free(p);
    }
    return hipSuccess;
}

inline hipError_t hipHostFree(void* p) { return hipFree(p); }
inline hipError_t hipMemset(void* p, int v, size_t s) { memset(p, v, s); return hipSuccess; }

// --- MOCK STREAMS ---
typedef int hipStream_t;
inline hipError_t hipStreamCreate(hipStream_t* s) { *s = 0; return hipSuccess; }
inline hipError_t hipStreamDestroy(hipStream_t s) { return hipSuccess; }
inline hipError_t hipStreamSynchronize(hipStream_t s) { return hipSuccess; }

// --- MOCK MEMCPY ---
enum hipMemcpyKind {
    hipMemcpyHostToDevice = 0,
    hipMemcpyDeviceToHost = 1,
    hipMemcpyDeviceToDevice = 2
};

// Async Copy (Just does a sync memcpy in mock)
inline hipError_t hipMemcpyAsync(void* dst, const void* src, size_t count, hipMemcpyKind kind, hipStream_t stream = 0) {
    memcpy(dst, src, count);
    return hipSuccess;
}

// --- LEAK CHECKER ---
inline void mock_check_leaks() {
    if (g_mock_total_bytes > 0 || !g_mock_allocations.empty()) {
        std::cerr << "[MOCK ERROR] Memory Leak Detected! Leaked Bytes: " << g_mock_total_bytes << std::endl;
        exit(1);
    }
}

// --- MOCK INTRINSICS ---
typedef float __half2;
inline float __float2half2_rn(float f) { return f; }
inline float __hfma2(float a, float b, float c) { return a*b+c; }
inline float __hneg2(float a) { return -a; }
inline float2 __half22float2(float a) { return {a, a}; }
inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return {x,y,z,w}; }
inline void atomicAdd(unsigned int* address, int val) { *address += val; }

// Math Intrinsics
inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
inline float __sinf(float x) { return sinf(x); }
inline float __cosf(float x) { return cosf(x); }
inline float __expf(float x) { return expf(x); }
inline float __logf(float x) { return logf(x); }

// --- AUTO-MAGIC LEAK CHECKER ---
struct MockLeakDetector {
    ~MockLeakDetector() { mock_check_leaks(); }
};
static MockLeakDetector g_leak_detector;

#endif
