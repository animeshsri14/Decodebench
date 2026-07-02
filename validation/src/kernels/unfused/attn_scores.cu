// attn_scores.cu — unfused attention scores kernel
// q·K^T → FP32 scores[H,L] MATERIALIZED IN GLOBAL MEMORY
// Layout: [H, L, D]
//
// Throughput-tuned: one warp per score element. Lanes read consecutive
// halves of the K row (fully coalesced 64B segments) and combine via
// shuffle reduction. The historical thread-per-score version made every
// thread walk its own 256B row, serializing on first-touch cache misses.
//
// Requires D % 64 == 0 (half2 loads, 32 lanes).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "decodebench_val/kernel_args.h"

namespace decodebench_val {
namespace kernels {
namespace unfused {

// grid = ceil(H * L / 8), block = 256 (8 warps, one score row per warp)
__global__ void attn_scores_kernel(KernelArgs args) {
  const int H    = args.H;
  const int L    = args.L;
  const int D    = args.D;
  const int gw   = (blockIdx.x * blockDim.x + threadIdx.x) / 32;  // global warp
  const int lane = threadIdx.x % 32;

  const int head = gw / L;
  const int l    = gw % L;
  if (head >= H) return;

  const __half2* q_row = reinterpret_cast<const __half2*>(args.q + head * D);
  const __half2* k_row = reinterpret_cast<const __half2*>(
      args.K + static_cast<size_t>(head) * L * D + static_cast<size_t>(l) * D);

  float dot = 0.0f;
  for (int i = lane; i < D / 2; i += 32) {
    const float2 qf = __half22float2(q_row[i]);
    const float2 kf = __half22float2(k_row[i]);
    dot += qf.x * kf.x + qf.y * kf.y;
  }
  for (int off = 16; off > 0; off >>= 1) {
    dot += __shfl_down_sync(0xffffffffu, dot, off);
  }

  if (lane == 0) {
    const float scale = rsqrtf(static_cast<float>(D));
    args.scores[head * L + l] = dot * scale;
  }
}

}  // namespace unfused
}  // namespace kernels
}  // namespace decodebench_val
