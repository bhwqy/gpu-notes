#pragma once
#include <cuda_runtime.h>

namespace qytools {

void launch_topk_int_kernel(
    int* input, int* topK, int* indices,
    int inputSliceSize,
    int outputSliceSize, // aka `k`
    int numInputSlices,
    cudaStream_t stream);

}

