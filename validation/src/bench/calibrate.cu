// calibrate.cu — DecodeBench calibration harness
// Commands:
//   calibrate --null               measure launch overhead (stream + graph)
//   calibrate --gate-g2 [--dim N]  compare unfused FP16 GEMV vs cuBLAS
// Uses CUDA events for all timing.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "decodebench_val/kernel_args.h"

// Forward-declare the gemv kernel from unfused/gemv.cu
namespace decodebench_val { namespace kernels { namespace unfused {
__global__ void gemv_kernel(KernelArgs args);
}}}

// Null kernel: empty body, 1 block
__global__ void null_kernel() {}

// ---- CUDA / cuBLAS error checking ----
#define CUDA_CHECK(call) do {                                    \
  cudaError_t _e = (call);                                       \
  if (_e != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
            __FILE__, __LINE__, cudaGetErrorString(_e));          \
    exit(1);                                                     \
  }                                                              \
} while (0)

#define CUBLAS_CHECK(call) do {                                  \
  cublasStatus_t _s = (call);                                    \
  if (_s != CUBLAS_STATUS_SUCCESS) {                             \
    fprintf(stderr, "cuBLAS error at %s:%d: %d\n",               \
            __FILE__, __LINE__, static_cast<int>(_s));            \
    exit(1);                                                     \
  }                                                              \
} while (0)

// ---- null calibration ----
static void run_null() {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Warmup
  null_kernel<<<1, 256, 0, stream>>>();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Measure stream launch: 500 iterations
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < 500; ++i) {
    null_kernel<<<1, 256, 0, stream>>>();
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms_stream;
  CUDA_CHECK(cudaEventElapsedTime(&ms_stream, start, stop));
  float t_stream = (ms_stream * 1000.0f) / 500.0f;

  // Measure graph launch
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  null_kernel<<<1, 256, 0, stream>>>();
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < 500; ++i) {
    CUDA_CHECK(cudaGraphLaunch(instance, stream));
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms_graph;
  CUDA_CHECK(cudaEventElapsedTime(&ms_graph, start, stop));
  float t_graph = (ms_graph * 1000.0f) / 500.0f;

  printf("t_launch_stream: %.2f us\n", t_stream);
  printf("t_launch_graph:  %.2f us\n", t_graph);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaGraphExecDestroy(instance));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---- G2 gate: unfused FP16 GEMV vs cuBLAS GEMV ----
// Uses cublasGemmEx to run FP16 GEMV (treat as m×1 GEMM).
// Row-major weight W[d_out][d_in], column vector x[d_in], output y[d_out].
//
// cuBLAS GEMM (column-major): C = op(A) * op(B)
// We want y = W * x (row-major semantics).
// Pass W as A with transa=CUBLAS_OP_T, lda=d_in:
//   A (pre-transpose) is d_in × d_out col-major stored at W with lda=d_in.
//   A[j][i] = W[i*d_in + j] = W_row[i][j]  ✓ (col-major index: j + i*lda)
//   op(A) = A^T is d_out × d_in, op(A)[i][j] = W_row[i][j]  ✓
//   m = d_out, n = 1, k = d_in
// B = x as d_in × 1 col-major, ldb = d_in
// C = y as d_out × 1 col-major, ldc = 1

static void run_gate_g2(int /*dim*/) {
  // Fixed realistic LLM FFN dimensions
  const int d_in  = 4096;
  const int d_out = 14336;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetStream(handle, stream));

  // Allocate
  size_t wbytes = static_cast<size_t>(d_out) * d_in * sizeof(__half);
  size_t xbytes = static_cast<size_t>(d_in) * sizeof(__half);
  size_t ybytes = static_cast<size_t>(d_out) * sizeof(__half);

  __half *d_W, *d_x, *d_y_cublas, *d_y_ours;
  CUDA_CHECK(cudaMalloc(&d_W, wbytes));
  CUDA_CHECK(cudaMalloc(&d_x, xbytes));
  CUDA_CHECK(cudaMalloc(&d_y_cublas, ybytes));
  CUDA_CHECK(cudaMalloc(&d_y_ours, ybytes));

  // Init host data
  srand(42);
  std::vector<__half> h_W(d_out * d_in);
  std::vector<__half> h_x(d_in);
  for (size_t i = 0; i < h_W.size(); ++i)
    h_W[i] = __float2half((static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * 0.02f);
  for (int i = 0; i < d_in; ++i)
    h_x[i] = __float2half(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);

  CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), wbytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), xbytes, cudaMemcpyHostToDevice));

  // Warmup both
  int grid = (d_out + 7) / 8;
  {
    decodebench_val::KernelArgs args;
    args.W = d_W; args.x = d_x; args.out = d_y_ours;
    args.d_in = d_in; args.d_out = d_out;
    decodebench_val::kernels::unfused::gemv_kernel<<<grid, 256, 0, stream>>>(args);
  }
  {
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      d_out, 1, d_in,
      &alpha, d_W, CUDA_R_16F, d_in,
      d_x, CUDA_R_16F, d_in,
      &beta, d_y_cublas, CUDA_R_16F, d_out,
      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Timing
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  int iters = 200;

  // --- Time our kernel ---
  {
    decodebench_val::KernelArgs args;
    args.W = d_W; args.x = d_x; args.out = d_y_ours;
    args.d_in = d_in; args.d_out = d_out;
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < iters; ++i)
      decodebench_val::kernels::unfused::gemv_kernel<<<grid, 256, 0, stream>>>(args);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
  }

  float ms_ours;
  CUDA_CHECK(cudaEventElapsedTime(&ms_ours, ev_start, ev_stop));
  float us_ours = (ms_ours * 1000.0f) / iters;

  // --- Time cuBLAS ---
  {
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < iters; ++i) {
      CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        d_out, 1, d_in,
        &alpha, d_W, CUDA_R_16F, d_in,
        d_x, CUDA_R_16F, d_in,
        &beta, d_y_cublas, CUDA_R_16F, d_out,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
  }

  float ms_cublas;
  CUDA_CHECK(cudaEventElapsedTime(&ms_cublas, ev_start, ev_stop));
  float us_cublas = (ms_cublas * 1000.0f) / iters;

  // Bandwidth: bytes read/written per invocation
  float bytes_per = 2.0f * static_cast<float>(
    static_cast<size_t>(d_out) * d_in + d_in + d_out);
  float bw_ours   = bytes_per / (us_ours * 1e3f);
  float bw_cublas = bytes_per / (us_cublas * 1e3f);

  printf("GEMV d_out=%d d_in=%d\n", d_out, d_in);
  printf("  Ours:   %.2f us  (%.2f GB/s)\n", us_ours, bw_ours);
  printf("  cuBLAS: %.2f us  (%.2f GB/s)\n", us_cublas, bw_cublas);

  float ratio = bw_ours / bw_cublas;
  printf("  Ratio (ours/cuBLAS): %.1f%%\n", ratio * 100.0f);
  printf("G2: %s\n", (ratio >= 0.90f) ? "PASS" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y_cublas)); CUDA_CHECK(cudaFree(d_y_ours));
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
  bool do_null   = false;
  bool do_gate_g2 = false;
  int  dim        = 4096;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--null") == 0)
      do_null = true;
    else if (strcmp(argv[i], "--gate-g2") == 0)
      do_gate_g2 = true;
    else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc)
      dim = atoi(argv[++i]);
  }

  if (!do_null && !do_gate_g2) {
    fprintf(stderr, "Usage: calibrate [--null] [--gate-g2] [--dim N]\n");
    return 1;
  }

  if (do_null)   run_null();
  if (do_gate_g2) run_gate_g2(dim);

  return 0;
}
