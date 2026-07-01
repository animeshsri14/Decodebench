// f2.cu — fused GEMV→SwiGLU kernel
// Each warp owns output element j: computes BOTH dot products (Wg row j, Wu row j).
// Then silu(g)*u in-register — no sync.
// Reads xh once for both dot products.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "decodebench_val/kernel_args.h"
#include "../device/warp_reduce.h"

namespace decodebench_val {
namespace kernels {
namespace fused {

// grid = ceil(d_out / 8), block = 256
__global__ void f2_kernel(KernelArgs args) {
  const int d_in         = args.d_in;
  const int d_out        = args.d_out;
  const int row          = blockIdx.x * 8 + (threadIdx.x / 32);
  const int lane         = threadIdx.x % 32;
  if (row >= d_out) return;

  const __half* Wg_row = args.Wg + row * d_in;
  const __half* Wu_row = args.Wu + row * d_in;

  // Warp-strided loop: 32 lanes × 8 elements = 256 elements per iteration.
  const int stride = 8 * 32;
  float acc_g = 0.0f;
  float acc_u = 0.0f;

  // Single pass over d_in: load xh once, compute both dot products
  for (int i = lane * 8; i < d_in; i += stride) {
    device::Vec8Half xv = device::load_vec8(args.xh + i);
    device::Vec8Half wgv = device::load_vec8(Wg_row + i);
    device::Vec8Half wuv = device::load_vec8(Wu_row + i);

    float x0 = __half2float(__low2half(xv.v0));
    float x1 = __half2float(__high2half(xv.v0));
    float x2 = __half2float(__low2half(xv.v1));
    float x3 = __half2float(__high2half(xv.v1));
    float x4 = __half2float(__low2half(xv.v2));
    float x5 = __half2float(__high2half(xv.v2));
    float x6 = __half2float(__low2half(xv.v3));
    float x7 = __half2float(__high2half(xv.v3));

    acc_g += __half2float(__low2half(wgv.v0))  * x0;
    acc_g += __half2float(__high2half(wgv.v0)) * x1;
    acc_g += __half2float(__low2half(wgv.v1))  * x2;
    acc_g += __half2float(__high2half(wgv.v1)) * x3;
    acc_g += __half2float(__low2half(wgv.v2))  * x4;
    acc_g += __half2float(__high2half(wgv.v2)) * x5;
    acc_g += __half2float(__low2half(wgv.v3))  * x6;
    acc_g += __half2float(__high2half(wgv.v3)) * x7;

    acc_u += __half2float(__low2half(wuv.v0))  * x0;
    acc_u += __half2float(__high2half(wuv.v0)) * x1;
    acc_u += __half2float(__low2half(wuv.v1))  * x2;
    acc_u += __half2float(__high2half(wuv.v1)) * x3;
    acc_u += __half2float(__low2half(wuv.v2))  * x4;
    acc_u += __half2float(__high2half(wuv.v2)) * x5;
    acc_u += __half2float(__low2half(wuv.v3))  * x6;
    acc_u += __half2float(__high2half(wuv.v3)) * x7;
  }

  // Warp reduction for both accumulators
  acc_g = device::warp_reduce_sum(acc_g);
  acc_u = device::warp_reduce_sum(acc_u);

  // Lane 0 computes silu(g) * u and writes
  if (lane == 0) {
    // silu(gate) = gate / (1 + exp(-gate))
    float silu_g = acc_g / (1.0f + expf(-acc_g));
    args.out[row] = __float2half(silu_g * acc_u);
  }
}

}  // namespace fused
}  // namespace kernels
}  // namespace decodebench_val
