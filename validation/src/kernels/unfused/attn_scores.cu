// attn_scores.cu — unfused attention scores kernel
// q·K^T → FP32 scores[H,L] MATERIALIZED IN GLOBAL MEMORY
// Layout: [H, L, D], L=1024
// One block per (head, L-element), each thread computes one score element.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "decodebench_val/kernel_args.h"

namespace decodebench_val {
namespace kernels {
namespace unfused {

// grid = H * ceil(L / 256), block = 256
__global__ void attn_scores_kernel(KernelArgs args) {
  const int H      = args.H;
  const int L      = args.L;
  const int D      = args.D;
  const int head   = blockIdx.x / ((L + blockDim.x - 1) / blockDim.x);
  const int l_base = (blockIdx.x % ((L + blockDim.x - 1) / blockDim.x)) * blockDim.x;
  const int l      = l_base + threadIdx.x;

  if (head >= H || l >= L) return;

  const __half* q_head = args.q + head * D;
  const __half* K_head_l = args.K + head * (L * D) + l * D;

  float dot = 0.0f;
  for (int d = 0; d < D; ++d) {
    dot += __half2float(q_head[d]) * __half2float(K_head_l[d]);
  }

  float scale = rsqrtf(static_cast<float>(D));
  args.scores[head * L + l] = dot * scale;
}

}  // namespace unfused
}  // namespace kernels
}  // namespace decodebench_val
