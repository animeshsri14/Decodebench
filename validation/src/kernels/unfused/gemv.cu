// gemv.cu — unfused GEMV kernel (warp-strided vectorized)
// One warp per output row; uint4 loads on weight; FP32 accumulation; shfl_xor reduce.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "decodebench_val/kernel_args.h"
#include "../device/warp_reduce.h"

namespace decodebench_val {
namespace kernels {
namespace unfused {

// grid = ceil(d_out / 8), block = 256 (8 warps)
__global__ void gemv_kernel(KernelArgs args) {
  const int d_in         = args.d_in;
  const int d_out        = args.d_out;
  const int row          = blockIdx.x * 8 + (threadIdx.x / 32);  // warp index
  const int lane         = threadIdx.x % 32;

  if (row >= d_out) return;

  const __half* weight_row = args.W + row * d_in;

  // Warp-strided loop: 32 lanes × 8 elements = 256 elements per iteration.
  const int stride = 8 * 32;
  float acc = 0.0f;

  for (int i = lane * 8; i < d_in; i += stride) {
    device::Vec8Half wv = device::load_vec8(weight_row + i);
    device::Vec8Half xv = device::load_vec8(args.x + i);

    acc += __half2float(__low2half(wv.v0))  * __half2float(__low2half(xv.v0));
    acc += __half2float(__high2half(wv.v0)) * __half2float(__high2half(xv.v0));
    acc += __half2float(__low2half(wv.v1))  * __half2float(__low2half(xv.v1));
    acc += __half2float(__high2half(wv.v1)) * __half2float(__high2half(xv.v1));
    acc += __half2float(__low2half(wv.v2))  * __half2float(__low2half(xv.v2));
    acc += __half2float(__high2half(wv.v2)) * __half2float(__high2half(xv.v2));
    acc += __half2float(__low2half(wv.v3))  * __half2float(__low2half(xv.v3));
    acc += __half2float(__high2half(wv.v3)) * __half2float(__high2half(xv.v3));
  }

  acc = device::warp_reduce_sum(acc);

  if (lane == 0) {
    args.out[row] = __float2half(acc);
  }
}

}  // namespace unfused
}  // namespace kernels
}  // namespace decodebench_val
