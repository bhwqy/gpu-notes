// 1. sorting + select
// 2. radix select + filter
// 3. warp select + block select
#include <cstdint>
#include <cuda_runtime.h>
#include "common.h"
#include "common.cuh"
#include "scan_utils.cuh"
#include "kernel.h"

namespace qytools {

template <typename scalar_t>
struct TopKTypeConfig {};

template <>
struct TopKTypeConfig<float> {
    typedef uint32_t RadixType;

    // Converts a float to an integer representation with the same
    // sorting; i.e., for floats f1, f2:
    // if f1 < f2 then convert(f1) < convert(f2)
    // We use this to enable radix selection of floating-point values.
    // This also gives a relative order for NaNs, but that's ok, as they
    // will all be adjacent
    // neg inf: signbit=1 exp=ff fraction=0 --> radix = 0 00 ff..
    // pos inf: signbit=0 exp=ff fraction=0 --> radix = 1 ff 00..
    // pos nan: signbit=0 exp=ff fraction>0 --> radix = 1 ff x>0
    // neg nan: signbit=1 exp=ff fraction>0 --> radix = 0 00 x<ff...
    static inline __device__ RadixType convert(float v) {
        RadixType x = __float_as_int(v);
        RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

        return (v == v) ? (x ^ mask) : 0xffffffff;
    }

    static inline __device__ float deconvert(RadixType v) {
        RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

        return __int_as_float(v ^ mask);
    }
};

template <>
struct TopKTypeConfig<uint8_t> {
    typedef uint32_t RadixType;

    static inline __device__ RadixType convert(uint8_t v) {
        return v;
    }

    static inline __device__ uint8_t deconvert(RadixType v) {
        return v;
    }
};

template <>
struct TopKTypeConfig<int8_t> {
    typedef uint32_t RadixType;

    static inline __device__ RadixType convert(int8_t v) {
        return 128u + v;
    }

    static inline __device__ int8_t deconvert(RadixType v) {
        return v - 128;
    }
};

template <>
struct TopKTypeConfig<int16_t> {
    typedef uint32_t RadixType;

    static inline __device__ RadixType convert(int16_t v) {
        static_assert(sizeof(short) == 2, "");
        return 32768u + v;
    }

    static inline __device__ int16_t deconvert(RadixType v) {
        return v - 32768;
    }
};

template <>
struct TopKTypeConfig<int32_t> {
    typedef uint32_t RadixType;

    static inline __device__ RadixType convert(int32_t v) {
        static_assert(sizeof(int) == 4, "");
        return 2147483648u + v;
    }

    static inline __device__ int32_t deconvert(RadixType v) {
        return v - 2147483648u;
    }
};

template <>
struct TopKTypeConfig<int64_t> {
    typedef uint64_t RadixType;

    static inline __device__ RadixType convert(int64_t v) {
        static_assert(sizeof(int64_t) == 8, "");
        return 9223372036854775808ull + v;
    }

    static inline __device__ int64_t deconvert(RadixType v) {
        return v - 9223372036854775808ull;
    }
};

template <>
struct TopKTypeConfig<double> {
    typedef uint64_t RadixType;

    static inline __device__ RadixType convert(double v) {
        RadixType x = __double_as_longlong(v);
        RadixType mask = -((x >> 63)) | 0x8000000000000000;
        return (v == v) ? (x ^ mask) : 0xffffffffffffffff;
    }

    static inline __device__ double deconvert(RadixType v) {
        RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
        return __longlong_as_double(v ^ mask);
    }
};

// This function counts the distribution of all input values in a
// slice we are selecting by radix digit at `radixDigitPos`, but only
// those that pass the filter `((v & desiredMask) == desired)`.
// This produces and broadcasts the seen counts for a single block only.
// `smem` must have at least `RadixSize` elements.
template <
    typename scalar_t,
    typename bitwise_t,
    typename index_t,
    typename CountType,
    int RadixSize,
    int RadixBits>
__device__ void countRadixUsingMask(
    CountType counts[RadixSize],
    CountType* smem,
    bitwise_t desired,
    bitwise_t desiredMask,
    int radixDigitPos,
    index_t sliceSize,
    const scalar_t* data) {
  // Clear out per-thread counts from a previous round
#pragma unroll
    for (int i = 0; i < RadixSize; ++i) {
        counts[i] = 0;
    }

    if (threadIdx.x < RadixSize) {
        smem[threadIdx.x] = 0;
    }
    __syncthreads();

    // Scan over all the data. Upon a read, the warp will accumulate
    // counts per each digit in the radix using warp voting.
    // Must be called outside of loop to ensure all threads participate
    unsigned mask = WARP_BALLOT(threadIdx.x < sliceSize);
    for (index_t i = threadIdx.x; i < sliceSize;) {
        bitwise_t val =
            TopKTypeConfig<scalar_t>::convert(doLdg(&data[i]));

        bool hasVal = ((val & desiredMask) == desired);
        bitwise_t digitInRadix = Bitfield<bitwise_t>::getBitfield(
            val, radixDigitPos, RadixBits);

#pragma unroll
        for (uint32_t j = 0; j < RadixSize; ++j) {
            bool vote = hasVal && (digitInRadix == j);
            counts[j] += __popc(WARP_BALLOT(vote, mask));
        }
        i += blockDim.x;
        mask = WARP_BALLOT(i < sliceSize, mask);
    }

    // Now, for each warp, sum values
    if (getLaneId() == 0) {
#pragma unroll
        for (uint32_t i = 0; i < RadixSize; ++i) {
            atomicAdd(&smem[i], counts[i]);
        }
    }

  __syncthreads();

  // For each thread, read in the total counts
#pragma unroll
    for (uint32_t i = 0; i < RadixSize; ++i) {
        counts[i] = smem[i];
    }

    __syncthreads();
}

// Over what radix we are selecting values
constexpr int RADIX_BITS = 2; // digits are base-(2 ^ RADIX_BITS)
constexpr int RADIX_SIZE = 4; // 2 ^ RADIX_BITS
constexpr int RADIX_MASK = (RADIX_SIZE - 1);

// This finds the unique value `v` that matches the pattern
// ((v & desired) == desiredMask) in our sorted int format
template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ scalar_t findPattern(
        scalar_t* smem,
        const scalar_t* data,
        index_t sliceSize,
        bitwise_t desired,
        bitwise_t desiredMask) {
    if (threadIdx.x < 2) {
        smem[threadIdx.x] = static_cast<scalar_t>(0);
    }
    __syncthreads();

    // All threads participate in the loop, in order to sync on the flag
    index_t numIterations =
        round_up(sliceSize, static_cast<index_t>(blockDim.x));
    for (index_t i = threadIdx.x; i < numIterations; i += blockDim.x) {
        bool inRange = (i < sliceSize);
        scalar_t v = inRange ? doLdg(&data[i]) : static_cast<scalar_t>(0);

        if (inRange &&
            ((TopKTypeConfig<scalar_t>::convert(v) & desiredMask) == desired)) {
            // There should not be conflicts if we are using findPattern,
            // since the result is unique
            smem[0] = static_cast<scalar_t>(1);
            smem[1] = v; // can't use val as the flag, since it could be 0
        }

        __syncthreads();

        scalar_t found = smem[0];
        scalar_t val = smem[1];

        __syncthreads();

        // Check to see if a thread found the value
        if (found != static_cast<scalar_t>(0)) {
            // all threads return this value
            return val;
        }
    }

    // should not get here
    return static_cast<scalar_t>(0);
}

// Returns the top-Kth element found in the data using radix selection
template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ void radixSelect(const scalar_t* data, index_t k,
        index_t sliceSize, int* smem, scalar_t* topK) {
    // Per-thread buckets into which we accumulate digit counts in our
    // radix
    int counts[RADIX_SIZE];

    // We only consider elements x such that (x & desiredMask) == desired
    // Initially, we consider all elements of the array, so the above
    // statement is true regardless of input.
    bitwise_t desired = 0;
    bitwise_t desiredMask = 0;

    // We are looking for the top kToFind-th element when iterating over
    // digits; this count gets reduced by elimination when counting
    // successive digits
    int kToFind = k;

    // We start at the most significant digit in our radix, scanning
    // through to the least significant digit
    for (int digitPos = sizeof(scalar_t) * 8 - RADIX_BITS; digitPos >= 0;
        digitPos -= RADIX_BITS) {
        // Count radix distribution for the current position and reduce
        // across all threads
        countRadixUsingMask<
            scalar_t,
            bitwise_t,
            index_t,
            int,
            RADIX_SIZE,
            RADIX_BITS>(
            counts,
            smem,
            desired,
            desiredMask,
            digitPos,
            sliceSize,
            data);

        auto found_unique = [&](int i, int count) -> bool {
            /* All threads have the same value in counts here, so all */
            /* threads will return from the function. */
            if (count == 1 && kToFind == 1) {
                /* There is a unique answer. */
                desired = Bitfield<bitwise_t>::setBitfield(
                    desired, i, digitPos, RADIX_BITS);
                desiredMask = Bitfield<bitwise_t>::setBitfield(
                    desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

                /* The answer is now the unique element v such that: */
                /* (v & desiredMask) == desired */
                /* However, we do not yet know what the actual element is. We */
                /* need to perform a search through the data to find the */
                /* element that matches this pattern. */
                *topK = findPattern<scalar_t, bitwise_t, index_t>(
                    (scalar_t*)smem,
                    data,
                    sliceSize,
                    desired,
                    desiredMask);
                return true;
            }
            return false;
        };
        auto found_non_unique = [&](int i, int count) -> bool {
            if (count >= kToFind) {
                desired =
                    Bitfield<bitwise_t>::setBitfield(
                        desired, i, digitPos, RADIX_BITS);
                desiredMask = Bitfield<bitwise_t>::setBitfield(
                    desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

                /* The top-Kth element v must now be one such that: */
                /* (v & desiredMask == desired) */
                /* but we haven't narrowed it down; we must check the next */
                /* least-significant digit */
                return true;
            }
            kToFind -= count;
            return false; // continue the loop
        };

        // All threads participate in the comparisons below to know the
        // final result
        // Process in ascending order
#pragma unroll
        for (int i = 0; i < RADIX_SIZE; ++i) {
            int count = counts[i];
            if (found_unique(i, count)) {
                return;
            }
            if (found_non_unique(i, count)) {
                break;
            }
        }
    } // end digitPos for

    // There is no unique result, but there is a non-unique result
    // matching `desired` exactly
    *topK = TopKTypeConfig<scalar_t>::deconvert(desired);
}

namespace sbtopk {

template <typename T>
struct AddOp {
    __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
        return (lhs + rhs);
    }
};

// 1024 threads per block
template <typename T, typename IndexType>
__global__ void gatherTopK(T* input, T* topK, IndexType* indices,
                           IndexType inputSliceSize,
                           IndexType outputSliceSize, // aka `k`
                           IndexType numInputSlices) {
    // Indices are limited to integer fp precision, so counts can fit in
    // int32, regardless of IndexType
    __shared__ int smem[32]; // one per each warp, up to warp limit
    IndexType slice = getLinearBlockId();
    if (slice >= numInputSlices) {
        return;
    }

    // Find the start offset for our slice
    const T* inputSliceStart = &input[slice];
    T* topKSliceStart = &topK[slice];
    IndexType* indicesSliceStart = &indices[slice];

    // Find the k-th highest element in our input
    T topKValue = static_cast<T>(0);
    radixSelect<T, typename TopKTypeConfig<T>::RadixType, IndexType>(
        inputSliceStart, outputSliceSize, inputSliceSize, smem, &topKValue);
    const auto topKConverted = TopKTypeConfig<T>::convert(topKValue);

    // Every value that is strictly less/greater than `pattern`
    // (depending on sort dir) in sorted int format is in the top-K.
    // The top-K value itself might not be unique.
    //
    // Since there are a variable number of elements that we see that
    // are within the top-k, we don't know at what index to write out
    // the resulting values.
    // In order to get this, we perform an exclusive prefix sum of
    // `hasTopK`. This will return the resulting index into which we
    // need to write the result, if a thread has a result.

    // All threads need to participate in the loop and the prefix sum,
    // but not necessarily in the load; hence loop bounds being rounded
    // up to a multiple of the block dim.
    IndexType numIterations = round_up(inputSliceSize, (IndexType) blockDim.x);
    IndexType writeIndexStart = 0;

    for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
        bool inRange = (i < inputSliceSize);
        T v =
        inRange ? doLdg(&inputSliceStart[i]) : static_cast<T>(0);
        const auto convertedV = TopKTypeConfig<T>::convert(v);
        bool hasTopK = inRange && (convertedV < topKConverted);
        int index;
        int carry;
        exclusiveBinaryPrefixScan<int, true>(smem, hasTopK, &index, &carry, AddOp<int>());

        if (hasTopK) {
            int writeIndex = writeIndexStart + index;

            IndexType topKOffset = writeIndex;
            IndexType indexOffset = writeIndex;

            topKSliceStart[topKOffset] = v;
            indicesSliceStart[indexOffset] = i;
        }

        writeIndexStart += carry;
    }

    // We need to fill in the rest with actual == top-K values.
    // The number that we need is outputSliceSize -
    // writeIndexStart. There might be more than that number available,
    // in which case we have to choose the first seen set. We do this
    // via a prefix sum to calculate indices for writing results.
    IndexType topKRemaining = (outputSliceSize - writeIndexStart);

    for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
        bool inRange = (i < inputSliceSize);
        T v = inRange ? doLdg(&inputSliceStart[i]) : static_cast<T>(0);
        const auto convertedV = TopKTypeConfig<T>::convert(v);
        bool hasTopK = inRange && (convertedV == topKConverted);

        int index;
        int carry;
        exclusiveBinaryPrefixScan<int, true>(
            smem, hasTopK, &index, &carry, AddOp<int>());

        if (hasTopK && index < topKRemaining) {
            int writeIndex = writeIndexStart + index;
            IndexType topKOffset = writeIndex;
            IndexType indexOffset = writeIndex;

            topKSliceStart[topKOffset] = v;
            indicesSliceStart[indexOffset] = i;
        }

        if (carry >= topKRemaining) {
            break;
        }

        topKRemaining -= carry;
        writeIndexStart += carry;
    }
};

}

void launch_topk_int_kernel(
        int* input, int* topK, int* indices,
        int inputSliceSize,
        int outputSliceSize, // aka `k`
        int numInputSlices,
        cudaStream_t stream) {

    dim3 grid;
    dim3 block(std::min(ceil_div(inputSliceSize, 32) * 32, 1024));
    sbtopk::gatherTopK<int, int><<<grid, block, 0, stream>>>(
        input, topK, indices,
        inputSliceSize,
        outputSliceSize,
        numInputSlices);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}

