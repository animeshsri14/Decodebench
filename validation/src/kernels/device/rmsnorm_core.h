#pragma once
// rmsnorm_core.h — reusable RMSNorm core: block-wide FP32 mean-square via warp reduce.
// Extracted from unfused/rmsnorm.cu per Phase 8c spec.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "warp_reduce.h"

namespace decodebench_val {
namespace device {

/// Compute inverse RMS for the given x array across the block.
/// All threads in the block call this; returns the same inv_rms for all.
/// smem: at least warp_count * sizeof(float) of shared memory.
__device__ __forceinline__ float block_inv_rms(
    const __half* x,           // input array
    int d,                     // length
    int tid,                   // threadIdx.x
    int warp_id,               // tid / 32
    int lane_id,               // tid % 32
    int block_dim,             // blockDim.x
    float* warp_sums_smem,     // shared mem scratch (at least 8 floats)
    float eps = 1e-5f
) {
  const int num_warps = block_dim / 32;

  // Sum of squares
  float sq_sum = 0.0f;
  for (int i = tid; i < d; i += block_dim) {
    float val = __half2float(x[i]);
    sq_sum += val * val;
  }

  sq_sum = block_reduce_sum_broadcast(sq_sum, warp_sums_smem, warp_id, lane_id, num_warps);
  return rsqrtf(sq_sum / static_cast<float>(d) + eps);
}

}  // namespace device
}  // namespace decodebench_val
