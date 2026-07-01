// swiglu.cu — unfused SwiGLU kernel
// Reads g[ff] + u[ff], writes silu(g) * u [ff].
// Grid = ceil(ff / 256), each thread handles one element.
// silu(x) = x / (1 + expf(-x)) computed in FP32.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "decodebench_val/kernel_args.h"

namespace decodebench_val {
namespace kernels {
namespace unfused {

__global__ void swiglu_kernel(KernelArgs args) {
  const int ff  = args.ff;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= ff) return;

  float gate = __half2float(args.g[idx]);
  float up   = __half2float(args.u[idx]);

  // silu(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
  float silu_gate = gate / (1.0f + expf(-gate));
  float result = silu_gate * up;

  args.out[idx] = __float2half(result);
}

}  // namespace unfused
}  // namespace kernels
}  // namespace decodebench_val
