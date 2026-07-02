// f1.cu — fused RMSNorm→GEMV kernel
// Each block redundantly re-reads x (8 KB, L2-resident) and computes RMS scale itself.
// No global sync between norm and GEMV.
// Same GEMV structure as unfused, but norm inline.
// Gamma is cooperatively loaded into shared memory exactly once per block.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "decodebench_val/kernel_args.h"
#include "../device/warp_reduce.h"

namespace decodebench_val {
namespace kernels {
namespace fused {

// grid = ceil(d_out / 8), block = 256
__global__ void f1_kernel(KernelArgs args) {
  const int d            = args.d;
  const int d_in         = args.d_in;
  const int d_out        = args.d_out;
  const int row          = blockIdx.x * 8 + (threadIdx.x / 32);
  const int lane         = threadIdx.x % 32;
  const int warp_id      = threadIdx.x / 32;
  // No early return here: every thread must reach __syncthreads() and the
  // block-wide reduction below, or a partial last block (d_out not a
  // multiple of 8) would diverge on the barriers. Row validity only gates
  // the per-row GEMV and the output write; the guard is warp-uniform
  // (row depends only on warp_id), so warp_reduce_sum stays full-warp.
  const bool row_valid   = (row < d_out);

  // --- Step 1: Load gamma into shared memory (d elements, <= 4096 = 8 KB) ---
  __shared__ __half gamma_shared[4096];  // max d
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    gamma_shared[i] = args.gamma[i];
  }
  __syncthreads();

  // --- Step 2: Compute RMS scale (redundant per block, L2-resident x) ---
  __shared__ float warp_sums[8];
  float sq_sum = 0.0f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float val = __half2float(args.x[i]);
    sq_sum += val * val;
  }
  sq_sum = device::block_reduce_sum_broadcast(sq_sum, warp_sums, warp_id, lane, blockDim.x / 32);
  float inv_rms = rsqrtf(sq_sum / static_cast<float>(d) + 1e-5f);

  // --- Step 3: GEMV with inline RMS normalization ---
  if (!row_valid) return;  // safe: no barriers below this point
  const __half* weight_row = args.W + row * d_in;
  // Warp-strided loop: 32 lanes × 8 elements = 256 elements per iteration.
  const int stride = 8 * 32;
  float acc = 0.0f;

  for (int i = lane * 8; i < d_in; i += stride) {
    device::Vec8Half wv = device::load_vec8(weight_row + i);
    device::Vec8Half xv = device::load_vec8(args.x + i);

    float g0 = __half2float(gamma_shared[i + 0]);
    float g1 = __half2float(gamma_shared[i + 1]);
    float g2 = __half2float(gamma_shared[i + 2]);
    float g3 = __half2float(gamma_shared[i + 3]);
    float g4 = __half2float(gamma_shared[i + 4]);
    float g5 = __half2float(gamma_shared[i + 5]);
    float g6 = __half2float(gamma_shared[i + 6]);
    float g7 = __half2float(gamma_shared[i + 7]);

    acc += __half2float(__low2half(wv.v0))  * __half2float(__low2half(xv.v0))  * g0 * inv_rms;
    acc += __half2float(__high2half(wv.v0)) * __half2float(__high2half(xv.v0)) * g1 * inv_rms;
    acc += __half2float(__low2half(wv.v1))  * __half2float(__low2half(xv.v1))  * g2 * inv_rms;
    acc += __half2float(__high2half(wv.v1)) * __half2float(__high2half(xv.v1)) * g3 * inv_rms;
    acc += __half2float(__low2half(wv.v2))  * __half2float(__low2half(xv.v2))  * g4 * inv_rms;
    acc += __half2float(__high2half(wv.v2)) * __half2float(__high2half(xv.v2)) * g5 * inv_rms;
    acc += __half2float(__low2half(wv.v3))  * __half2float(__low2half(xv.v3))  * g6 * inv_rms;
    acc += __half2float(__high2half(wv.v3)) * __half2float(__high2half(xv.v3)) * g7 * inv_rms;
  }

  acc = device::warp_reduce_sum(acc);

  if (lane == 0) {
    args.out[row] = __float2half(acc);
  }
}

}  // namespace fused
}  // namespace kernels
}  // namespace decodebench_val
