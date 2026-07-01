#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK_TIMING(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(_e)); \
} while(0)

/// CUDA event-based wall-clock timer as specified in Phase 5d.
struct CudaEventTimer {
    cudaEvent_t start, stop;

    CudaEventTimer() {
        CUDA_CHECK_TIMING(cudaEventCreate(&start));
        CUDA_CHECK_TIMING(cudaEventCreate(&stop));
    }

    ~CudaEventTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Non-copyable, non-movable
    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void record_start(cudaStream_t s = 0) {
        CUDA_CHECK_TIMING(cudaEventRecord(start, s));
    }

    void record_stop(cudaStream_t s = 0) {
        CUDA_CHECK_TIMING(cudaEventRecord(stop, s));
    }

    float elapsed_ms() {
        CUDA_CHECK_TIMING(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK_TIMING(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};
#endif
