#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace qytools {

constexpr int WARP_SIZE = 32;

// Bitfield
template <typename T>
struct Bitfield {};

template <>
struct Bitfield<unsigned int> {
    static __device__ __host__ __forceinline__
    unsigned int getBitfield(unsigned int val, int pos, int len) {
        unsigned int ret;
        asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
        return ret;
    }

    static __device__ __host__ __forceinline__
    unsigned int setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
        unsigned int ret;
        asm("bfi.b32 %0, %1, %2, %3, %4;" :
            "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
        return ret;
    }
};

template <>
struct Bitfield<uint64_t> {
    static __device__ __host__ __forceinline__
    uint64_t getBitfield(uint64_t val, int pos, int len) {
        uint64_t ret;
        asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
        return ret;
    }

    static __device__ __host__ __forceinline__
    uint64_t setBitfield(uint64_t val, uint64_t toInsert, int pos, int len) {
        uint64_t ret;
        asm("bfi.b64 %0, %1, %2, %3, %4;" :
            "=l"(ret) : "l"(toInsert), "l"(val), "r"(pos), "r"(len));
        return ret;
    }
};

// lane
__device__ __forceinline__ int getLaneId() {
    int laneId;
    asm("mov.s32 %0, %%laneid;" : "=r"(laneId) );
    return laneId;
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskLe() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGt() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}

__forceinline__ __device__ uint getLinearBlockId() {
    return blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
}

// warp
__device__ __forceinline__ unsigned int ACTIVE_MASK() {
    return __activemask();
}

__device__ __forceinline__ void WARP_SYNC(unsigned mask = 0xffffffff) {
    return __syncwarp(mask);
}

__device__ __forceinline__ unsigned int WARP_BALLOT(int predicate, unsigned int mask = 0xffffffff) {
    return __ballot_sync(mask, predicate);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff) {
    return __shfl_xor_sync(mask, value, laneMask, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff) {
    return __shfl_sync(mask, value, srcLane, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_UP(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff) {
    return __shfl_up_sync(mask, value, delta, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff) {
    return __shfl_down_sync(mask, value, delta, width);
}

template <typename T>
__device__ __forceinline__ T doLdg(const T* p) {
    return __ldg(p);
}

// 
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
__forceinline__ __host__ __device__ T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

/**
   Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
   multiple of b
*/
template <typename T>
__forceinline__ __host__ __device__ T round_up(T a, T b) {
    return ceil_div(a, b) * b;
}

}

