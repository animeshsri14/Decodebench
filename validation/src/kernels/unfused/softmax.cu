// softmax.cu — unfused row softmax over L, FP32 in/out
// One block per row (per head): subtract max, exp, normalize.

#include <cuda_runtime.h>

#include "decodebench_val/kernel_args.h"
#include "../device/warp_reduce.h"

namespace decodebench_val {
namespace kernels {
namespace unfused {

// grid = H, block = 256
__global__ void softmax_kernel(KernelArgs args) {
  const int H         = args.H;
  const int L         = args.L;
  const int head      = blockIdx.x;
  const int tid       = threadIdx.x;
  const int warp_id   = tid / 32;
  const int lane_id   = tid % 32;
  const int num_warps = blockDim.x / 32;

  if (head >= H) return;

  float* score_row = args.scores + head * L;
  float* prob_row  = args.probs + head * L;

  __shared__ float warp_maxs[8];
  __shared__ float warp_sums[8];

  // Step 1: find row max
  float row_max = -INFINITY;
  for (int i = tid; i < L; i += blockDim.x) {
    float s = score_row[i];
    if (s > row_max) row_max = s;
  }
  row_max = device::block_reduce_max_broadcast(row_max, warp_maxs, warp_id, lane_id, num_warps);

  // Step 2: compute exp and sum
  float row_sum = 0.0f;
  for (int i = tid; i < L; i += blockDim.x) {
    float v = expf(score_row[i] - row_max);
    prob_row[i] = v;  // store exp temporarily
    row_sum += v;
  }
  row_sum = device::block_reduce_sum_broadcast(row_sum, warp_sums, warp_id, lane_id, num_warps);

  // Step 3: normalize
  float inv_sum = 1.0f / (row_sum + 1e-8f);
  for (int i = tid; i < L; i += blockDim.x) {
    prob_row[i] *= inv_sum;
  }
}

}  // namespace unfused
}  // namespace kernels
}  // namespace decodebench_val
