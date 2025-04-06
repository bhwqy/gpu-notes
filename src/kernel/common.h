#pragma once
#include <glog/logging.h>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(ret) \
    do { \
        cudaError_t error = ret; \
        CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)
#endif

#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
    classname(const classname&) = delete;\
    classname& operator=(const classname&) = delete;

