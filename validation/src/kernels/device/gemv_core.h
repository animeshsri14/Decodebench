#pragma once
// gemv_core.h — reusable GEMV core: warp-strided vectorized loop with uint4 loads.
// Extracted from unfused/gemv.cu per Phase 8b spec.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "decodebench_val/kernel_args.h"
#include "warp_reduce.h"

namespace decodebench_val {
namespace device {

/// GEMV core: one warp per output row, FP32 accumulation, 128-bit vectorized loads.
/// Shared by unfused and fused kernels (F1, F2).
__device__ __forceinline__ float gemv_core_dot(
    const __half* weight_row,   // start of weight row (row-major)
    const __half* x_ptr,        // input vector
    int d_in,                   // length of input vector
    int lane,                   // thread lane in warp (0..31)
    int num_warps               // total number of warps across grid
) {
  const int stride = 8 * num_warps;
  float acc = 0.0f;

  for (int i = lane * 8; i < d_in; i += stride) {
    Vec8Half wv = load_vec8(weight_row + i);
    Vec8Half xv = load_vec8(x_ptr + i);

    acc += __half2float(__low2half(wv.v0))  * __half2float(__low2half(xv.v0));
    acc += __half2float(__high2half(wv.v0)) * __half2float(__high2half(xv.v0));
    acc += __half2float(__low2half(wv.v1))  * __half2float(__low2half(xv.v1));
    acc += __half2float(__high2half(wv.v1)) * __half2float(__high2half(xv.v1));
    acc += __half2float(__low2half(wv.v2))  * __half2float(__low2half(xv.v2));
    acc += __half2float(__high2half(wv.v2)) * __half2float(__high2half(xv.v2));
    acc += __half2float(__low2half(wv.v3))  * __half2float(__low2half(xv.v3));
    acc += __half2float(__high2half(wv.v3)) * __half2float(__high2half(xv.v3));
  }

  return warp_reduce_sum(acc);
}

}  // namespace device
}  // namespace decodebench_val
