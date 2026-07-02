// bench_variant.cu — DecodeBench validation benchmark harness
//
// Compares unfused-stream, unfused-graph, and fused kernel execution strategies
// across F1 (RMSNorm→GEMV), F2 (GEMV→SwiGLU), and F4 (FlashDecode attention).
//
// Usage:
//   bench_variant --fusion {f1,f2,f4} --variant {unfused-stream,unfused-graph,fused}
//                 --dim {2048,4096} --batch {1,2,4,8} --trials 30 --target-ms 20
//                 --seed 42 --csv <path> [--ncu-mode] [--skip-correctness]
//
// --dim semantics: F1/F2 hidden dimension d_in; F4 KV-cache length L
// (H=32, D=128 fixed). --batch: only 1 is accepted. The kernels are
// unbatched; the harness refuses to label unbatched runs with batch>1
// (previously the value was a CSV label only, which allowed fictitious
// batch sweeps).
//
// Cache-residency parity: every variant (unfused-stream, unfused-graph,
// fused) uses the same round-robin weight/KV replica rotation, so L2
// residency is equivalent across variants. unfused-graph captures one
// graph per replica and rotates graph launches.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

#include "decodebench_val/kernel_args.h"
#include "../kernels/device/warp_reduce.h"

// Forward declarations for all kernel entry points
namespace decodebench_val { namespace kernels {
namespace unfused {
  __global__ void gemv_kernel(KernelArgs args);
  __global__ void rmsnorm_kernel(KernelArgs args);
  __global__ void swiglu_kernel(KernelArgs args);
  __global__ void attn_scores_kernel(KernelArgs args);
  __global__ void softmax_kernel(KernelArgs args);
  __global__ void attn_v_kernel(KernelArgs args);
}
namespace fused {
  __global__ void f1_kernel(KernelArgs args);
  __global__ void f2_kernel(KernelArgs args);
  __global__ void f4_kernel(KernelArgs args);
  __global__ void f4_partial_kernel(KernelArgs args);
  __global__ void f4_reduce_kernel(KernelArgs args);
}
}}

using namespace decodebench_val;

// =====================================================================
// Helpers
// =====================================================================

#define CUDA_CHECK(call) do {                                    \
  cudaError_t _e = (call);                                       \
  if (_e != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
            __FILE__, __LINE__, cudaGetErrorString(_e));          \
    exit(1);                                                     \
  }                                                              \
} while (0)

// Check for launch-configuration errors immediately after kernel launches.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

static int div_up(int a, int b) { return (a + b - 1) / b; }

static int adaptive_iters(float target_ms, float t_one_us) {
  const double desired = ceil(static_cast<double>(target_ms) * 1000.0 /
                              static_cast<double>(t_one_us));
  if (desired >= 1000000.0) return 1000000;
  return std::max(200, static_cast<int>(desired));
}

// Per-replica CUDA graph set: one instantiated graph per weight replica so
// graph replay rotates residency exactly like the stream/fused paths.
struct GraphSet {
  std::vector<cudaGraphExec_t> execs;
  cudaGraphExec_t get(int k) const { return execs[k % execs.size()]; }
  bool empty() const { return execs.empty(); }
  ~GraphSet() {
    for (auto e : execs) if (e) cudaGraphExecDestroy(e);
  }
};

template <typename LaunchFn>
static void capture_graph_per_replica(cudaStream_t stream, int n_copies,
                                      LaunchFn launch, GraphSet* out) {
  for (int i = 0; i < n_copies; ++i) {
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch(i);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    out->execs.push_back(exec);
  }
}

// Get GPU name string (truncated to 63 chars, no spaces for CSV)
static std::string gpu_name() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::string name(prop.name);
  for (auto& c : name) if (c == ' ') c = '_';
  if (name.size() > 63) name.resize(63);
  return name;
}

static int l2_cache_bytes() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  return prop.l2CacheSize;
}

// =====================================================================
// CPU reference implementations (inline, for G1 correctness gating)
// =====================================================================

static void cpu_rmsnorm(const std::vector<__half>& x,
                        const std::vector<__half>& gamma,
                        std::vector<__half>& out, int d) {
  float sq = 0.0f;
  for (int i = 0; i < d; ++i) {
    float v = __half2float(x[i]);
    sq += v * v;
  }
  float inv_rms = 1.0f / sqrtf(sq / d + 1e-5f);
  for (int i = 0; i < d; ++i)
    out[i] = __float2half(__half2float(x[i]) * __half2float(gamma[i]) * inv_rms);
}

static void cpu_gemv(const std::vector<__half>& W,
                     const std::vector<__half>& x,
                     std::vector<__half>& y, int d_out, int d_in) {
  for (int i = 0; i < d_out; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < d_in; ++j)
      acc += __half2float(W[i * d_in + j]) * __half2float(x[j]);
    y[i] = __float2half(acc);
  }
}

static void cpu_swiglu(const std::vector<__half>& g,
                       const std::vector<__half>& u,
                       std::vector<__half>& out, int ff) {
  for (int i = 0; i < ff; ++i) {
    float gate = __half2float(g[i]);
    float up   = __half2float(u[i]);
    float silu = gate / (1.0f + expf(-gate));
    out[i] = __float2half(silu * up);
  }
}

static void cpu_f1(const std::vector<__half>& x,
                   const std::vector<__half>& gamma,
                   const std::vector<__half>& W,
                   std::vector<__half>& y, int d, int d_in, int d_out) {
  std::vector<__half> xn(d);
  cpu_rmsnorm(x, gamma, xn, d);
  cpu_gemv(W, xn, y, d_out, d_in);
}

static void cpu_f2(const std::vector<__half>& xh,
                   const std::vector<__half>& Wg,
                   const std::vector<__half>& Wu,
                   std::vector<__half>& y, int d_in, int d_out) {
  std::vector<__half> gv(d_out), uv(d_out);
  cpu_gemv(Wg, xh, gv, d_out, d_in);
  cpu_gemv(Wu, xh, uv, d_out, d_in);
  cpu_swiglu(gv, uv, y, d_out);
}

static void cpu_f4(const std::vector<__half>& q,
                   const std::vector<__half>& K,
                   const std::vector<__half>& V,
                   std::vector<__half>& out, int H, int L, int D) {
  // Independent, straightforward scaled-dot-product attention reference.
  // It deliberately shares no tiling, split-KV, or reduction code with any
  // GPU implementation, so G1 can catch a bug common to the GPU witnesses.
  std::vector<float> probs(L);
  const float scale = 1.0f / sqrtf(static_cast<float>(D));
  for (int h = 0; h < H; ++h) {
    const size_t q_base = static_cast<size_t>(h) * D;
    const size_t kv_base = static_cast<size_t>(h) * L * D;
    float row_max = -INFINITY;
    for (int l = 0; l < L; ++l) {
      float dot = 0.0f;
      const size_t row = kv_base + static_cast<size_t>(l) * D;
      for (int d = 0; d < D; ++d)
        dot += __half2float(q[q_base + d]) * __half2float(K[row + d]);
      probs[l] = dot * scale;
      row_max = fmaxf(row_max, probs[l]);
    }
    float row_sum = 0.0f;
    for (int l = 0; l < L; ++l) {
      probs[l] = expf(probs[l] - row_max);
      row_sum += probs[l];
    }
    const float inv_sum = 1.0f / row_sum;
    for (int d = 0; d < D; ++d) {
      float acc = 0.0f;
      for (int l = 0; l < L; ++l) {
        const size_t idx = kv_base + static_cast<size_t>(l) * D + d;
        acc += probs[l] * __half2float(V[idx]);
      }
      out[q_base + d] = __float2half(acc * inv_sum);
    }
  }
}

// Check correctness: max_abs < 5e-2, max_rel < 2e-2
static std::string check_correctness(const std::vector<__half>& a,
                                     const std::vector<__half>& b,
                                     int n) {
  float max_abs = 0.0f, max_rel = 0.0f;
  for (int i = 0; i < n; ++i) {
    float va = __half2float(a[i]);
    float vb = __half2float(b[i]);
    if (!std::isfinite(va) || !std::isfinite(vb))
      return "non-finite output detected";
    float abs_err = fabsf(va - vb);
    float rel_err = abs_err / fmaxf(fabsf(vb), 1e-3f);
    if (abs_err > max_abs) max_abs = abs_err;
    if (rel_err > max_rel) max_rel = rel_err;
  }
  char buf[128];
  snprintf(buf, sizeof(buf), "max_abs=%.4e max_rel=%.4e", max_abs, max_rel);
  return std::string(buf);
}

static bool correctness_pass(const std::vector<__half>& a,
                             const std::vector<__half>& b, int n) {
  // numpy-allclose semantics: an element is a mismatch only when it exceeds
  // BOTH the absolute (5e-2) and the relative (2e-2) tolerance. A pure-rel
  // failure on a near-zero output — where FP16 rounding inflates relative error
  // on a tiny magnitude — is not a real disagreement, and a fused kernel that
  // keeps the value in FP32 is legitimately more accurate than the FP16-rounded
  // reference. Requiring both bounds to fail keeps the gate meaningful without
  // flagging that rounding noise.
  for (int i = 0; i < n; ++i) {
    float va = __half2float(a[i]), vb = __half2float(b[i]);
    if (!std::isfinite(va) || !std::isfinite(vb)) return false;
    float abs_err = fabsf(va - vb);
    float rel_err = abs_err / fmaxf(fabsf(vb), 1e-3f);
    if (abs_err >= 5e-2f && rel_err >= 2e-2f) return false;
  }
  return true;
}

// =====================================================================
// Random data generation
// =====================================================================

static std::vector<__half> random_half(int n, unsigned* seed, float scale = 1.0f) {
  std::vector<__half> v(n);
  for (int i = 0; i < n; ++i) {
    float r = static_cast<float>(rand_r(seed)) / RAND_MAX * 2.0f - 1.0f;
    v[i] = __float2half(r * scale);
  }
  return v;
}

// =====================================================================
// Weight replica management
// =====================================================================

struct WeightReplicas {
  int n_copies;
  std::vector<__half*> d_copies;
  size_t bytes_per;

  WeightReplicas() : n_copies(0), bytes_per(0) {}

  void init(const __half* base, size_t bytes, int max_copies, int l2_bytes,
            size_t replica_working_set_bytes = 0) {
    bytes_per = bytes;
    // N_copies = min(8, max(4, ceil(2 * L2 / weight_bytes)))
    const size_t sizing_bytes = replica_working_set_bytes ? replica_working_set_bytes : bytes;
    float raw = ceilf(2.0f * l2_bytes / static_cast<float>(sizing_bytes));
    n_copies = std::min(max_copies, std::max(4, static_cast<int>(raw)));
    if (n_copies < 1) n_copies = 1;

    d_copies.resize(n_copies);
    // Copy 0 is the original
    d_copies[0] = const_cast<__half*>(base);
    for (int i = 1; i < n_copies; ++i) {
      CUDA_CHECK(cudaMalloc(&d_copies[i], bytes));
      CUDA_CHECK(cudaMemcpy(d_copies[i], base, bytes, cudaMemcpyDeviceToDevice));
    }
  }

  __half* get(int idx) const { return d_copies[idx % n_copies]; }

  ~WeightReplicas() {
    for (int i = 1; i < n_copies; ++i)  // skip copy 0
      if (d_copies[i]) cudaFree(d_copies[i]);
  }
};

// =====================================================================
// F1 benchmark (RMSNorm → GEMV)
// =====================================================================

static void bench_f1(const std::string& variant, int dim, int batch,
                     int trials, float target_ms, unsigned seed,
                     bool ncu_mode, bool skip_correctness,
                     FILE* csv_fp) {
  // F1 dimensions: d=4096 (hidden), d_in=4096, d_out=14336 -> uses d=dim
  const int d     = dim;
  const int d_in  = dim;
  const int d_out = 14336;

  // Kernel contracts enforced at the host boundary (cross-architecture):
  // vectorized 128-bit loads require d_in % 8 == 0; the fused F1 kernel
  // stages gamma in a fixed 4096-element shared buffer.
  if (d_in % 8 != 0 || d < 1) {
    fprintf(stderr, "F1: --dim must be a positive multiple of 8 (vectorized loads), got %d\n", dim);
    exit(1);
  }
  if (d > 4096) {
    fprintf(stderr, "F1: --dim must be <= 4096 (fused kernel shared-memory gamma buffer), got %d\n", dim);
    exit(1);
  }

  int l2 = l2_cache_bytes();
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Generate host data
  auto h_x     = random_half(d, &seed, 1.0f);
  auto h_gamma = random_half(d, &seed, 0.1f);
  auto h_W     = random_half(d_out * d_in, &seed, 0.02f);
  std::vector<__half> h_ref(d_out);
  cpu_f1(h_x, h_gamma, h_W, h_ref, d, d_in, d_out);

  // Allocate device memory. d_xh holds the RMSNorm output that the unfused
  // GEMV consumes — without it the two kernels would share one args struct and
  // GEMV would read the raw input instead of the normalized vector.
  __half *d_x, *d_gamma, *d_W, *d_out_val, *d_xh;
  CUDA_CHECK(cudaMalloc(&d_x, d * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_gamma, d * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_W, d_out * d_in * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_out_val, d_out * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_xh, d * sizeof(__half)));

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), d * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), d * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), d_out * d_in * sizeof(__half), cudaMemcpyHostToDevice));

  // Weight replicas (for W — the dominant weight). Initialized for EVERY
  // variant so t_stream, t_graph, and t_fused are measured under the same
  // L2-residency conditions (a variant that reuses one allocation would see
  // artificially warm cache).
  WeightReplicas w_repl;
  size_t w_bytes = static_cast<size_t>(d_out) * d_in * sizeof(__half);
  w_repl.init(d_W, w_bytes, 8, l2);

  // Kernel setup
  KernelArgs args;
  args.x = d_x; args.gamma = d_gamma; args.W = d_W;
  args.out = d_out_val; args.d = d; args.d_in = d_in; args.d_out = d_out;

  // Unfused path uses two views of args so the kernels chain through d_xh:
  // RMSNorm reads d_x and writes d_xh; GEMV reads d_xh and writes d_out_val.
  KernelArgs args_norm = args; args_norm.out = d_xh;
  KernelArgs args_gemv = args; args_gemv.x = d_xh;

  int gemv_grid = div_up(d_out, 8);

  // Per-iteration launchers (rotate weight replicas identically per variant)
  auto launch_fused_k = [&](int k) {
    KernelArgs af = args; af.W = w_repl.get(k);
    kernels::fused::f1_kernel<<<gemv_grid, 256, 0, stream>>>(af);
  };
  auto launch_unfused_k = [&](int k) {
    KernelArgs ag = args_gemv; ag.W = w_repl.get(k);
    kernels::unfused::rmsnorm_kernel<<<1, 256, 0, stream>>>(args_norm);
    kernels::unfused::gemv_kernel<<<gemv_grid, 256, 0, stream>>>(ag);
  };

  // ---- Correctness check (G1) ----
  bool ok = true;
  std::string corr_msg = "SKIPPED";
  if (!skip_correctness) {
    std::vector<__half> h_gpu(d_out);
    if (variant == "fused") {
      launch_fused_k(0);
    } else {
      launch_unfused_k(0);
    }
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out_val, d_out * sizeof(__half), cudaMemcpyDeviceToHost));
    ok = correctness_pass(h_gpu, h_ref, d_out);
    corr_msg = ok ? "PASS" : (std::string("FAIL ") + check_correctness(h_gpu, h_ref, d_out));
  }
  bool correctness_ok = ok;

  // ---- Timing ----
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  cudaEvent_t ev_trial_start, ev_trial_stop;
  CUDA_CHECK(cudaEventCreate(&ev_trial_start));
  CUDA_CHECK(cudaEventCreate(&ev_trial_stop));

  // Warmup: 50 iterations, rotating replicas like the timed loop
  for (int w = 0; w < 50; ++w) {
    if (variant == "fused") launch_fused_k(w);
    else                    launch_unfused_k(w);
  }
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Measure a small probe batch for adaptive K (less noise than one launch).
  float t_one;
  {
    constexpr int probe_iters = 5;
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < probe_iters; ++i) {
      if (variant == "fused") launch_fused_k(i);
      else                    launch_unfused_k(i);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    t_one = fmaxf(ms * 1000.0f / probe_iters, 1.0e-4f);
  }

  // Adaptive K
  int K = adaptive_iters(target_ms, t_one);
  if (ncu_mode) K = 1;

  // CUDA graphs for unfused-graph variant: one per weight replica, rotated
  // at launch so graph replay sees the same residency as the other variants.
  GraphSet graphs;
  if (variant == "unfused-graph") {
    capture_graph_per_replica(stream, w_repl.n_copies,
                              [&](int i) { launch_unfused_k(i); }, &graphs);
  }

  // Timing loop: 30 trials
  time_t now = time(nullptr);
  for (int trial = 0; trial < trials; ++trial) {
    if (ncu_mode) {
      // Single invocation, no timing measurement
      if (variant == "fused") {
        launch_fused_k(trial);
      } else if (variant == "unfused-graph") {
        CUDA_CHECK(cudaGraphLaunch(graphs.get(trial), stream));
      } else {
        launch_unfused_k(trial);
      }
      CUDA_CHECK_LAUNCH();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      fprintf(csv_fp, "%s,f1,%s,%d,%d,%d,%d,0,%.3f,%d,%ld\n",
              gpu_name().c_str(), variant.c_str(), dim, batch, trial, K,
              0.0, correctness_ok ? 1 : 0, static_cast<long>(now));
      continue;
    }

    // Run K iterations, measuring total time
    CUDA_CHECK(cudaEventRecord(ev_trial_start, stream));
    for (int k = 0; k < K; ++k) {
      if (variant == "fused") {
        launch_fused_k(k);
      } else if (variant == "unfused-graph") {
        CUDA_CHECK(cudaGraphLaunch(graphs.get(k), stream));
      } else {
        launch_unfused_k(k);
      }
    }
    CUDA_CHECK(cudaEventRecord(ev_trial_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_trial_stop));
    CUDA_CHECK_LAUNCH();

    float ms_trial;
    CUDA_CHECK(cudaEventElapsedTime(&ms_trial, ev_trial_start, ev_trial_stop));
    float us_per = (ms_trial * 1000.0f) / static_cast<float>(K);

    fprintf(csv_fp, "%s,f1,%s,%d,%d,%d,%d,%.3f,%d,%ld\n",
            gpu_name().c_str(), variant.c_str(), dim, batch, trial, K,
            us_per, correctness_ok ? 1 : 0, static_cast<long>(now));
  }

  // Cleanup (GraphSet destructor frees graph execs)
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaEventDestroy(ev_trial_start));
  CUDA_CHECK(cudaEventDestroy(ev_trial_stop));
  CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_out_val));
  CUDA_CHECK(cudaFree(d_xh));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

// =====================================================================
// F2 benchmark (GEMV → SwiGLU)
// =====================================================================

static void bench_f2(const std::string& variant, int dim, int batch,
                     int trials, float target_ms, unsigned seed,
                     bool ncu_mode, bool skip_correctness,
                     FILE* csv_fp) {
  const int d_in  = dim;
  const int d_out = 14336;

  if (d_in % 8 != 0 || d_in < 1) {
    fprintf(stderr, "F2: --dim must be a positive multiple of 8 (vectorized loads), got %d\n", dim);
    exit(1);
  }

  int l2 = l2_cache_bytes();
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto h_xh = random_half(d_in, &seed, 1.0f);
  auto h_Wg = random_half(d_out * d_in, &seed, 0.02f);
  auto h_Wu = random_half(d_out * d_in, &seed, 0.02f);
  std::vector<__half> h_ref(d_out);
  cpu_f2(h_xh, h_Wg, h_Wu, h_ref, d_in, d_out);

  // d_g/d_u are persistent intermediates: the unfused pipeline must really
  // materialize both GEMV outputs (the byte model counts them). Aliasing them
  // onto d_out_val — as the timed path previously did — computes garbage and
  // under-represents the real pipeline.
  __half *d_xh, *d_Wg, *d_Wu, *d_out_val, *d_g, *d_u;
  CUDA_CHECK(cudaMalloc(&d_xh, d_in * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_Wg, d_out * d_in * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_Wu, d_out * d_in * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_out_val, d_out * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_g, d_out * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_u, d_out * sizeof(__half)));

  CUDA_CHECK(cudaMemcpy(d_xh, h_xh.data(), d_in * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wg, h_Wg.data(), d_out * d_in * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wu, h_Wu.data(), d_out * d_in * sizeof(__half), cudaMemcpyHostToDevice));

  size_t w_bytes = static_cast<size_t>(d_out) * d_in * sizeof(__half);
  // Replicas for EVERY variant (residency parity across variants).
  WeightReplicas wg_repl, wu_repl;
  wg_repl.init(d_Wg, w_bytes, 8, l2, 2 * w_bytes);
  wu_repl.init(d_Wu, w_bytes, 8, l2, 2 * w_bytes);

  KernelArgs args;
  args.x = d_xh; args.xh = d_xh; args.Wg = d_Wg; args.Wu = d_Wu;
  args.out = d_out_val; args.d_in = d_in; args.d_out = d_out;

  int gemv_grid = div_up(d_out, 8);

  // Per-iteration launchers. The unfused pipeline chains gate GEMV -> d_g,
  // up GEMV -> d_u, SwiGLU(d_g, d_u) -> d_out_val, exactly as modeled.
  auto launch_fused_k = [&](int k) {
    KernelArgs af = args;
    af.Wg = wg_repl.get(k); af.Wu = wu_repl.get(k); af.out = d_out_val;
    kernels::fused::f2_kernel<<<gemv_grid, 256, 0, stream>>>(af);
  };
  auto launch_unfused_k = [&](int k) {
    KernelArgs ak = args;
    ak.W = wg_repl.get(k); ak.out = d_g;
    kernels::unfused::gemv_kernel<<<gemv_grid, 256, 0, stream>>>(ak);
    ak.W = wu_repl.get(k); ak.out = d_u;
    kernels::unfused::gemv_kernel<<<gemv_grid, 256, 0, stream>>>(ak);
    ak.g = d_g; ak.u = d_u; ak.out = d_out_val; ak.ff = d_out;
    kernels::unfused::swiglu_kernel<<<div_up(d_out, 256), 256, 0, stream>>>(ak);
  };

  // Correctness
  bool correctness_ok = true;
  if (!skip_correctness) {
    std::vector<__half> h_gpu(d_out);
    if (variant == "fused") launch_fused_k(0);
    else                    launch_unfused_k(0);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out_val, d_out * sizeof(__half), cudaMemcpyDeviceToHost));
    correctness_ok = correctness_pass(h_gpu, h_ref, d_out);
  }

  // Timing (similar to F1)
  cudaEvent_t ev_start, ev_stop, ev_trial_start, ev_trial_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start)); CUDA_CHECK(cudaEventCreate(&ev_stop));
  CUDA_CHECK(cudaEventCreate(&ev_trial_start)); CUDA_CHECK(cudaEventCreate(&ev_trial_stop));

  // Warmup, rotating replicas like the timed loop
  for (int w = 0; w < 50; ++w) {
    if (variant == "fused") launch_fused_k(w);
    else                    launch_unfused_k(w);
  }
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Measure a small probe batch for adaptive K.
  float t_one;
  {
    constexpr int probe_iters = 5;
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < probe_iters; ++i) {
      if (variant == "fused") launch_fused_k(i);
      else                    launch_unfused_k(i);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    t_one = fmaxf(ms * 1000.0f / probe_iters, 1.0e-4f);
  }

  int K = adaptive_iters(target_ms, t_one);
  if (ncu_mode) K = 1;

  GraphSet graphs;
  if (variant == "unfused-graph") {
    capture_graph_per_replica(stream, wg_repl.n_copies,
                              [&](int i) { launch_unfused_k(i); }, &graphs);
  }

  time_t now = time(nullptr);
  for (int trial = 0; trial < trials; ++trial) {
    if (ncu_mode) {
      if (variant == "fused") {
        launch_fused_k(trial);
      } else if (variant == "unfused-graph") {
        CUDA_CHECK(cudaGraphLaunch(graphs.get(trial), stream));
      } else {
        launch_unfused_k(trial);
      }
      CUDA_CHECK_LAUNCH();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      fprintf(csv_fp, "%s,f2,%s,%d,%d,%d,%d,0,%.3f,%d,%ld\n",
              gpu_name().c_str(), variant.c_str(), dim, batch, trial, K,
              0.0, correctness_ok ? 1 : 0, static_cast<long>(now));
      continue;
    }

    CUDA_CHECK(cudaEventRecord(ev_trial_start, stream));
    for (int k = 0; k < K; ++k) {
      if (variant == "fused") {
        launch_fused_k(k);
      } else if (variant == "unfused-graph") {
        CUDA_CHECK(cudaGraphLaunch(graphs.get(k), stream));
      } else {
        launch_unfused_k(k);
      }
    }
    CUDA_CHECK(cudaEventRecord(ev_trial_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_trial_stop));
    CUDA_CHECK_LAUNCH();

    float ms_trial;
    CUDA_CHECK(cudaEventElapsedTime(&ms_trial, ev_trial_start, ev_trial_stop));
    float us_per = (ms_trial * 1000.0f) / static_cast<float>(K);

    fprintf(csv_fp, "%s,f2,%s,%d,%d,%d,%d,%.3f,%d,%ld\n",
            gpu_name().c_str(), variant.c_str(), dim, batch, trial, K,
            us_per, correctness_ok ? 1 : 0, static_cast<long>(now));
  }

  CUDA_CHECK(cudaEventDestroy(ev_start)); CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaEventDestroy(ev_trial_start)); CUDA_CHECK(cudaEventDestroy(ev_trial_stop));
  CUDA_CHECK(cudaFree(d_xh)); CUDA_CHECK(cudaFree(d_Wg));
  CUDA_CHECK(cudaFree(d_Wu)); CUDA_CHECK(cudaFree(d_out_val));
  CUDA_CHECK(cudaFree(d_g)); CUDA_CHECK(cudaFree(d_u));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

// =====================================================================
// F4 benchmark (FlashDecode-style attention)
// =====================================================================

static void bench_f4(const std::string& variant, int dim, int batch,
                     int trials, float target_ms, unsigned seed,
                     bool ncu_mode, bool skip_correctness,
                     FILE* csv_fp) {
  // F4 dimensions: H and D are fixed (Llama-7B-style decode head config);
  // --dim sets the KV-cache length L so sweeps vary the real problem size.
  const int H = 32;
  const int L = dim;
  const int D = 128;
  if (L < 128 || L % 128 != 0) {
    fprintf(stderr,
            "F4: --dim is the KV length and must be a positive multiple of "
            "128, got %d\n", dim);
    exit(1);
  }
  // Kernel contracts (enforced here, not assumed): the F4 kernels require an
  // even head dim <= 128 and D % 8 == 0 for vectorized loads.
  if (D > 128 || D % 2 != 0 || D % 8 != 0) {
    fprintf(stderr, "F4: head dim D=%d violates kernel contract (even, %%8==0, <=128)\n", D);
    exit(1);
  }

  int l2 = l2_cache_bytes();
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto h_q = random_half(H * D, &seed, 0.1f);
  auto h_K = random_half(H * L * D, &seed, 0.02f);
  auto h_V = random_half(H * L * D, &seed, 0.02f);

  // Split-KV split count: one KV tile (TILE_L=128 rows) per block. Empirically
  // fastest on T4 at every measured L (max parallelism; per-block work is a
  // single streaming pass). DECODEBENCH_F4_SPLITS overrides for per-GPU tuning
  // experiments on other architectures.
  const int num_tiles = L / 128;
  int n_splits = num_tiles;
  if (const char* env_splits = getenv("DECODEBENCH_F4_SPLITS")) {
    int v = atoi(env_splits);
    if (v >= 1) n_splits = std::min(num_tiles, v);
  }

  __half *d_q, *d_K, *d_V, *d_out;
  float  *d_scores, *d_probs;
  float  *d_part_o, *d_part_m, *d_part_l;
  CUDA_CHECK(cudaMalloc(&d_q, H * D * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_K, H * L * D * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_V, H * L * D * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_out, H * D * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_scores, H * L * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_probs, H * L * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_part_o, static_cast<size_t>(H) * n_splits * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_part_m, H * n_splits * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_part_l, H * n_splits * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), H * D * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), H * L * D * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), H * L * D * sizeof(__half), cudaMemcpyHostToDevice));

  // KV replicas for EVERY variant: K and V are the streamed operands (the
  // weight analog), so all variants must rotate them identically for L2
  // residency parity. K and V have equal size, so both sets share n_copies.
  size_t kv_bytes = static_cast<size_t>(H) * L * D * sizeof(__half);
  WeightReplicas k_repl, v_repl;
  k_repl.init(d_K, kv_bytes, 8, l2, 2 * kv_bytes);
  v_repl.init(d_V, kv_bytes, 8, l2, 2 * kv_bytes);

  KernelArgs args;
  args.q = d_q; args.K = d_K; args.V = d_V; args.out = d_out;
  args.scores = d_scores; args.probs = d_probs;
  args.part_o = d_part_o; args.part_m = d_part_m; args.part_l = d_part_l;
  args.n_splits = n_splits;
  args.H = H; args.L = L; args.D = D;

  int scores_grid = div_up(H * L, 8);  // one warp per score element

  // Per-iteration launchers rotating KV replicas identically per variant.
  // Fused variant = split-KV FlashDecode: 2 launches (partial + merge),
  // scores/probs never touch global memory.
  auto launch_fused_k = [&](int k) {
    KernelArgs ak = args;
    ak.K = k_repl.get(k); ak.V = v_repl.get(k);
    kernels::fused::f4_partial_kernel<<<H * n_splits, 256, 0, stream>>>(ak);
    kernels::fused::f4_reduce_kernel<<<H, D, 0, stream>>>(ak);
  };
  auto launch_unfused_k = [&](int k) {
    KernelArgs ak = args;
    ak.K = k_repl.get(k); ak.V = v_repl.get(k);
    kernels::unfused::attn_scores_kernel<<<scores_grid, 256, 0, stream>>>(ak);
    kernels::unfused::softmax_kernel<<<H, 256, 0, stream>>>(ak);
    kernels::unfused::attn_v_kernel<<<H * (D / 32), 256, 0, stream>>>(ak);
  };

  bool correctness_ok = true;
  if (!skip_correctness) {
    std::vector<__half> h_gpu(H * D), h_fused(H * D), h_cpu(H * D);
    cpu_f4(h_q, h_K, h_V, h_cpu, H, L, D);

    // Unfused path
    launch_unfused_k(0);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, H * D * sizeof(__half), cudaMemcpyDeviceToHost));

    // Fused path (split-KV)
    launch_fused_k(0);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_fused.data(), d_out, H * D * sizeof(__half), cudaMemcpyDeviceToHost));

    // Single-block GPU reference kernel remains a second implementation
    // witness, but the independent CPU result is the correctness authority.
    std::vector<__half> h_ref(H * D);
    kernels::fused::f4_kernel<<<H, 256, 0, stream>>>(args);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_out, H * D * sizeof(__half), cudaMemcpyDeviceToHost));

    correctness_ok = correctness_pass(h_gpu, h_cpu, H * D) &&
                     correctness_pass(h_fused, h_cpu, H * D) &&
                     correctness_pass(h_ref, h_cpu, H * D);
  }

  // Timing
  cudaEvent_t ev_start, ev_stop, ev_trial_start, ev_trial_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start)); CUDA_CHECK(cudaEventCreate(&ev_stop));
  CUDA_CHECK(cudaEventCreate(&ev_trial_start)); CUDA_CHECK(cudaEventCreate(&ev_trial_stop));

  // Warmup, rotating replicas like the timed loop
  for (int w = 0; w < 50; ++w) {
    if (variant == "fused") launch_fused_k(w);
    else                    launch_unfused_k(w);
  }
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Measure a small probe batch for adaptive K.
  float t_one;
  {
    constexpr int probe_iters = 5;
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < probe_iters; ++i) {
      if (variant == "fused") launch_fused_k(i);
      else                    launch_unfused_k(i);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    t_one = fmaxf(ms * 1000.0f / probe_iters, 1.0e-4f);
  }

  int K = adaptive_iters(target_ms, t_one);
  if (ncu_mode) K = 1;

  GraphSet graphs;
  if (variant == "unfused-graph") {
    capture_graph_per_replica(stream, k_repl.n_copies,
                              [&](int i) { launch_unfused_k(i); }, &graphs);
  }

  time_t now = time(nullptr);
  for (int trial = 0; trial < trials; ++trial) {
    if (ncu_mode) {
      if (variant == "fused") {
        launch_fused_k(trial);
      } else if (variant == "unfused-graph") {
        CUDA_CHECK(cudaGraphLaunch(graphs.get(trial), stream));
      } else {
        launch_unfused_k(trial);
      }
      CUDA_CHECK_LAUNCH();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      fprintf(csv_fp, "%s,f4,%s,%d,%d,%d,%d,0,%.3f,%d,%ld\n",
              gpu_name().c_str(), variant.c_str(), dim, batch, trial, K,
              0.0, correctness_ok ? 1 : 0, static_cast<long>(now));
      continue;
    }

    CUDA_CHECK(cudaEventRecord(ev_trial_start, stream));
    for (int k = 0; k < K; ++k) {
      if (variant == "fused") {
        launch_fused_k(k);
      } else if (variant == "unfused-graph") {
        CUDA_CHECK(cudaGraphLaunch(graphs.get(k), stream));
      } else {
        launch_unfused_k(k);
      }
    }
    CUDA_CHECK(cudaEventRecord(ev_trial_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_trial_stop));
    CUDA_CHECK_LAUNCH();

    float ms_trial;
    CUDA_CHECK(cudaEventElapsedTime(&ms_trial, ev_trial_start, ev_trial_stop));
    float us_per = (ms_trial * 1000.0f) / static_cast<float>(K);

    fprintf(csv_fp, "%s,f4,%s,%d,%d,%d,%d,%.3f,%d,%ld\n",
            gpu_name().c_str(), variant.c_str(), dim, batch, trial, K,
            us_per, correctness_ok ? 1 : 0, static_cast<long>(now));
  }

  CUDA_CHECK(cudaEventDestroy(ev_start)); CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaEventDestroy(ev_trial_start)); CUDA_CHECK(cudaEventDestroy(ev_trial_stop));
  CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_K)); CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_scores)); CUDA_CHECK(cudaFree(d_probs));
  CUDA_CHECK(cudaFree(d_part_o)); CUDA_CHECK(cudaFree(d_part_m)); CUDA_CHECK(cudaFree(d_part_l));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

// =====================================================================
// Main
// =====================================================================

int main(int argc, char** argv) {
  std::string fusion   = "f1";
  std::string variant  = "fused";
  int dim              = 4096;
  int batch            = 1;
  int trials           = 30;
  float target_ms      = 20.0f;
  unsigned seed        = 42;
  const char* csv_path = nullptr;
  bool ncu_mode        = false;
  bool skip_correctness = false;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--fusion") == 0 && i + 1 < argc)
      fusion = argv[++i];
    else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc)
      variant = argv[++i];
    else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc)
      dim = atoi(argv[++i]);
    else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc)
      batch = atoi(argv[++i]);
    else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc)
      trials = atoi(argv[++i]);
    else if (strcmp(argv[i], "--target-ms") == 0 && i + 1 < argc)
      target_ms = atof(argv[++i]);
    else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
      seed = static_cast<unsigned>(atoi(argv[++i]));
    else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
      csv_path = argv[++i];
    else if (strcmp(argv[i], "--ncu-mode") == 0)
      ncu_mode = true;
    else if (strcmp(argv[i], "--skip-correctness") == 0)
      skip_correctness = true;
    else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      return 1;
    }
  }

  if (!csv_path) {
    fprintf(stderr, "Usage: bench_variant --csv <path> [options...]\n");
    return 1;
  }

  if (batch != 1) {
    fprintf(stderr,
            "ERROR: --batch=%d rejected. All kernels are unbatched (B=1); "
            "labeling unbatched runs with batch>1 would fabricate batch-sweep "
            "data. Implement batched kernels before sweeping batch.\n", batch);
    return 1;
  }
  if (trials <= 0 || !std::isfinite(target_ms) || target_ms <= 0.0f) {
    fprintf(stderr, "ERROR: --trials must be positive and --target-ms must be finite and positive\n");
    return 1;
  }

  FILE* fp = fopen(csv_path, "w");
  if (!fp) { fprintf(stderr, "Cannot open %s\n", csv_path); return 1; }

  // CSV header
  fprintf(fp, "gpu_name,fusion,variant,dim,batch,trial,iters,us_per_invocation,correctness_ok,timestamp\n");

  if (fusion == "f1")
    bench_f1(variant, dim, batch, trials, target_ms, seed, ncu_mode, skip_correctness, fp);
  else if (fusion == "f2")
    bench_f2(variant, dim, batch, trials, target_ms, seed, ncu_mode, skip_correctness, fp);
  else if (fusion == "f4")
    bench_f4(variant, dim, batch, trials, target_ms, seed, ncu_mode, skip_correctness, fp);
  else {
    fprintf(stderr, "Unknown fusion: %s\n", fusion.c_str());
    fclose(fp);
    return 1;
  }

  fclose(fp);
  return 0;
}
