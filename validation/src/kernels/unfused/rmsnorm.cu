// rmsnorm.cu — unfused RMSNorm kernel
// Reads x[d] + gamma[d], writes normalized output.
// Block-wide FP32 mean-square via warp reduce; grid = 1, 256 threads.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "decodebench_val/kernel_args.h"
#include "../device/warp_reduce.h"

namespace decodebench_val {
namespace kernels {
namespace unfused {

__global__ void rmsnorm_kernel(KernelArgs args) {
  const int d           = args.d;
  const int tid         = threadIdx.x;
  const int warp_id     = tid / 32;
  const int lane_id     = tid % 32;
  const int num_warps   = blockDim.x / 32;  // 8

  __shared__ float warp_sums[8];  // one per warp

  // Step 1: compute sum of squares (each thread covers strided elements)
  float sq_sum = 0.0f;
  for (int i = tid; i < d; i += blockDim.x) {
    float val = __half2float(args.x[i]);
    sq_sum += val * val;
  }

  // Block-wide reduction: sum of squares → mean square → rms
  sq_sum = device::block_reduce_sum_broadcast(sq_sum, warp_sums, warp_id, lane_id, num_warps);
  float rms = sqrtf(sq_sum / static_cast<float>(d) + 1e-5f);
  float inv_rms = 1.0f / rms;

  // Step 2: normalize and scale by gamma
  for (int i = tid; i < d; i += blockDim.x) {
    float val = __half2float(args.x[i]);
    float gam = __half2float(args.gamma[i]);
    args.out[i] = __float2half(val * inv_rms * gam);
  }
}

}  // namespace unfused
}  // namespace kernels
}  // namespace decodebench_val
