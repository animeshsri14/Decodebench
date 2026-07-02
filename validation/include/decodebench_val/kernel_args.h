#pragma once
// kernel_args.h — unified kernel argument struct for DecodeBench validation kernels
// Spec-compliant: dual-mode for CUDA (#ifdef __CUDACC__) and CPU-only builds.

#include <cstdint>
#include <cstddef>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

namespace decodebench_val {

// Dimension tags for fused kernel dispatch
enum class FusionKind : int {
  F1_RMSNORM_GEMV = 1,
  F2_GEMV_SWIGLU  = 2,
  F4_FLASHDECODE  = 4,
};

// Single struct covering all kernel variants.
// Unused fields should be zero/nullptr — kernels ignore them.
// CPU-side uses uint16_t mirror so CPU TUs compile without CUDA headers.
struct KernelArgs {
  // --- Common I/O (FP16) ---
#ifdef __CUDACC__
  const __half* x;           // [d_in]         input activation / RMSNorm input
  const __half* gamma;       // [d]            RMSNorm weight
  const __half* W;           // [d_out, d_in]  GEMV weight matrix (row-major)
  __half* out;               // [d_out]        generic output
#else
  const uint16_t* x;
  const uint16_t* gamma;
  const uint16_t* W;
  uint16_t* out;
#endif
  int d_in;                  // input (hidden) dimension for GEMV
  int d_out;                 // output dimension for GEMV
  int batch;                 // batch size (unused in single-decode kernels; reserved)

  // --- F4 attention fields (FP16) ---
#ifdef __CUDACC__
  const __half* q;           // [n_heads, head_dim]   attention query
  const __half* K;           // [n_heads, seq_len, head_dim]  keys
  const __half* V;           // [n_heads, seq_len, head_dim]  values
#else
  const uint16_t* q;
  const uint16_t* K;
  const uint16_t* V;
#endif
  float* scores;             // [n_heads, seq_len]  FP32 attention scores (unfused only)
  int n_heads;               // number of attention heads
  int seq_len;               // KV-cache sequence length
  int head_dim;              // head dimension

  int weight_replica_stride; // round-robin offset for weight replicas (0 = unused)

  // --- Additional fields used by unfused kernels ---
#ifdef __CUDACC__
  const __half* g;           // [ff]            SwiGLU gate projection
  const __half* u;           // [ff]            SwiGLU up-projection
  const __half* xh;          // [d_in]          F2: shared input for both dot products
  const __half* Wg;          // [d_out, d_in]   F2: gate weight matrix
  const __half* Wu;          // [d_out, d_in]   F2: up weight matrix
#else
  const uint16_t* g;
  const uint16_t* u;
  const uint16_t* xh;
  const uint16_t* Wg;
  const uint16_t* Wu;
#endif
  float* probs;              // [n_heads, seq_len]  FP32 softmax output (unfused only)

  // --- F4 split-KV FlashDecode scratch (fused variant only) ---
  float* part_o;             // [n_heads, n_splits, head_dim]  unnormalized partial outputs
  float* part_m;             // [n_heads, n_splits]            per-split running max
  float* part_l;             // [n_heads, n_splits]            per-split exp-sum
  int n_splits;              // KV splits per head (1 = no split)

  // --- Convenience dimension aliases (set to same values as spec fields) ---
  int d;                     // RMSNorm dimension (= d_in typically)
  int H;                     // heads (= n_heads)
  int L;                     // seq_len
  int D;                     // head_dim
  int ff;                    // FFN intermediate dimension (= d_out typically)

  // --- Fused dispatch tag ---
  FusionKind fusion;

  // Zero-initialize
  KernelArgs()
    : x(nullptr), gamma(nullptr), W(nullptr), out(nullptr),
      d_in(0), d_out(0), batch(0),
      q(nullptr), K(nullptr), V(nullptr),
      scores(nullptr),
      n_heads(0), seq_len(0), head_dim(0),
      weight_replica_stride(0),
      g(nullptr), u(nullptr), xh(nullptr), Wg(nullptr), Wu(nullptr),
      probs(nullptr),
      part_o(nullptr), part_m(nullptr), part_l(nullptr), n_splits(0),
      d(0), H(0), L(0), D(0), ff(0),
      fusion(FusionKind::F1_RMSNORM_GEMV) {}
};

}  // namespace decodebench_val
