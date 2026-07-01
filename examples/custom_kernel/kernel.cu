#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ── Replace these three kernels with your own ────────────────────────────────

__global__ void scale_kernel(const __half* x, const __half* w, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hmul(x[i], w[i]);
}

__global__ void bias_kernel(const __half* x, const __half* b, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hadd(x[i], b[i]);
}

__global__ void scale_bias_fused_kernel(const __half* x, const __half* w,
                                         const __half* b, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hadd(__hmul(x[i], w[i]), b[i]);
}

// ── Launch wrappers (called from bindings.cpp) ────────────────────────────────

static int blocks(int n) { return (n + 255) / 256; }

void scale_launch(const __half* x, const __half* w, __half* out, int n, cudaStream_t s) {
    scale_kernel<<<blocks(n), 256, 0, s>>>(x, w, out, n);
}

void bias_launch(const __half* x, const __half* b, __half* out, int n, cudaStream_t s) {
    bias_kernel<<<blocks(n), 256, 0, s>>>(x, b, out, n);
}

void scale_bias_fused_launch(const __half* x, const __half* w, const __half* b,
                              __half* out, int n, cudaStream_t s) {
    scale_bias_fused_kernel<<<blocks(n), 256, 0, s>>>(x, w, b, out, n);
}
