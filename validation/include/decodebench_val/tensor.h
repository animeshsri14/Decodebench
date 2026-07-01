#pragma once

#include <cstddef>
#include <stdexcept>
#include <cstring>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(_e)); \
} while(0)
#else
#define CUDA_CHECK(call) ((void)0)
#endif

/// RAII host/device buffer wrapper as specified in Phase 5c.
template<typename T>
struct DeviceBuffer {
    T* data = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;

    /// Allocate on device if CUDA available, else on host.
    void alloc(size_t n) {
        free_buf();
        count = n;
        if (n == 0) return;
#ifdef __CUDACC__
        CUDA_CHECK(cudaMalloc(&data, n * sizeof(T)));
#else
        data = new T[n];
#endif
    }

    /// Copy from host to device buffer.
    void copy_from(const T* host_src) {
        if (!host_src || count == 0) return;
#ifdef __CUDACC__
        CUDA_CHECK(cudaMemcpy(data, host_src, count * sizeof(T), cudaMemcpyHostToDevice));
#else
        std::memcpy(data, host_src, count * sizeof(T));
#endif
    }

    /// Copy from device buffer to host.
    void copy_to(T* host_dst) const {
        if (!host_dst || count == 0) return;
#ifdef __CUDACC__
        CUDA_CHECK(cudaMemcpy(host_dst, data, count * sizeof(T), cudaMemcpyDeviceToHost));
#else
        std::memcpy(host_dst, data, count * sizeof(T));
#endif
    }

    /// Free the buffer.
    void free_buf() {
        if (!data) return;
#ifdef __CUDACC__
        CUDA_CHECK(cudaFree(data));
#else
        delete[] data;
#endif
        data = nullptr;
        count = 0;
    }

    ~DeviceBuffer() { free_buf(); }

    // Non-copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Movable
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data(other.data), count(other.count) {
        other.data = nullptr;
        other.count = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free_buf();
            data = other.data;
            count = other.count;
            other.data = nullptr;
            other.count = 0;
        }
        return *this;
    }
};
