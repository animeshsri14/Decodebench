// attn_v.cu — unfused attention·V kernel
// probs·V → out[H,D] FP16
// One block per (head, d_out element), dot product over L.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "decodebench_val/kernel_args.h"
#include "../device/warp_reduce.h"

namespace decodebench_val {
namespace kernels {
namespace unfused {

// grid = H * D, block = 256
// Each block computes one output element: out[head][dim] = sum_l prob[head][l] * V[head][l][dim]
__global__ void attn_v_kernel(KernelArgs args) {
  const int H          = args.H;
  const int L          = args.L;
  const int D          = args.D;
  const int head       = blockIdx.x / D;
  const int dim        = blockIdx.x % D;
  const int tid        = threadIdx.x;
  const int warp_id    = tid / 32;
  const int lane_id    = tid % 32;

  if (head >= H) return;

  const float* prob_row = args.probs + head * L;
  const __half* V_head  = args.V + head * (L * D);

  float acc = 0.0f;
  for (int l = tid; l < L; l += blockDim.x) {
    acc += prob_row[l] * __half2float(V_head[l * D + dim]);
  }

  __shared__ float warp_acc[8];
  acc = device::block_reduce_sum(acc, warp_acc, warp_id, lane_id, blockDim.x / 32);

  if (tid == 0) {
    args.out[head * D + dim] = __float2half(acc);
  }
}

}  // namespace unfused
}  // namespace kernels
}  // namespace decodebench_val
