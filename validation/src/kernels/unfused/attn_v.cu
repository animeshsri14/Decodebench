// attn_v.cu — unfused attention·V kernel
// probs·V → out[H,D] FP16
//
// Throughput-tuned: one block per (head, group of DV output dims). Each block
// streams its V sub-columns row-wise through shared memory with vectorized
// float4 loads (fully coalesced, unlike the historical one-block-per-element
// version that read V at a 2-byte-per-256-byte stride). This keeps the
// unfused baseline byte-efficiency comparable to the fused split-KV kernel so
// the fusion comparison isolates eliminated bytes + launches, not kernel
// tuning quality.
//
// Requires: L % ROWS == 0 (L is a multiple of 128), D % DV == 0.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "decodebench_val/kernel_args.h"

namespace decodebench_val {
namespace kernels {
namespace unfused {

static constexpr int DV   = 32;   // output dims per block
static constexpr int ROWS = 128;  // V rows staged per tile

// grid = H * (D / DV), block = 256
__global__ void attn_v_kernel(KernelArgs args) {
  const int H      = args.H;
  const int L      = args.L;
  const int D      = args.D;
  const int groups = D / DV;
  const int head   = blockIdx.x / groups;
  const int dg     = blockIdx.x % groups;
  const int tid    = threadIdx.x;

  if (head >= H) return;

  const float*  prob_row = args.probs + head * L;
  const __half* V_head   = args.V + static_cast<size_t>(head) * L * D + dg * DV;

  // Thread t accumulates output dim d = t % DV over row-group t / DV.
  const int d      = tid % DV;
  const int rgroup = tid / DV;
  constexpr int RG = 256 / DV;         // row-groups per block
  constexpr int VEC_PER_ROW = DV / 8;  // float4 loads per staged row (64B)

  __shared__ __half V_tile[ROWS][DV];
  __shared__ float  P_tile[ROWS];
  __shared__ float  red[256];

  float acc = 0.0f;
  for (int l0 = 0; l0 < L; l0 += ROWS) {
    __syncthreads();
    for (int i = tid; i < ROWS; i += blockDim.x) {
      P_tile[i] = prob_row[l0 + i];
    }
    float4* dst = reinterpret_cast<float4*>(&V_tile[0][0]);
    for (int i = tid; i < ROWS * VEC_PER_ROW; i += blockDim.x) {
      const int r = i / VEC_PER_ROW;
      const int v = i % VEC_PER_ROW;
      const float4* src =
          reinterpret_cast<const float4*>(V_head + static_cast<size_t>(l0 + r) * D);
      dst[i] = src[v];
    }
    __syncthreads();

    for (int r = rgroup; r < ROWS; r += RG) {
      acc += P_tile[r] * __half2float(V_tile[r][d]);
    }
  }

  red[tid] = acc;
  __syncthreads();
  if (tid < DV) {
    float s = 0.0f;
    for (int g = 0; g < RG; ++g) s += red[g * DV + tid];
    args.out[head * D + dg * DV + tid] = __float2half(s);
  }
}

}  // namespace unfused
}  // namespace kernels
}  // namespace decodebench_val
