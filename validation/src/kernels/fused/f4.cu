// f4.cu — FlashDecode-style fused attention kernel
// One block per (batch, head). Online softmax over L-tiles in smem/registers.
// Scores NEVER written to global memory — realizes the eliminated ~3%.
// Tiled loop over L dimension.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "decodebench_val/kernel_args.h"
#include "../device/warp_reduce.h"

namespace decodebench_val {
namespace kernels {
namespace fused {

// Tile size for L dimension (must divide L evenly; L=1024 → TILE_L=128)
static constexpr int TILE_L = 128;

// grid = H, block = 256
// D = 128 fits in registers per thread
__global__ void f4_kernel(KernelArgs args) {
  const int H       = args.H;
  const int L       = args.L;
  const int D       = args.D;
  const int head    = blockIdx.x;
  const int tid     = threadIdx.x;
  const int lane    = tid % 32;
  const int warp_id = tid / 32;

  if (head >= H) return;

  const __half* q_head = args.q + head * D;
  const __half* K_head = args.K + head * (L * D);
  const __half* V_head = args.V + head * (L * D);

  // Scale factor: 1/sqrt(D)
  const float scale = rsqrtf(static_cast<float>(D));

  // --- Preload query into registers ---
  // D=128, 256 threads → each thread loads D/256*2 = 1 half (using half2 for 2 at a time)
  // Actually D=128, we need each thread to load part of q.
  // Simpler: each thread loads strided elements, then we'll use shared memory or broadcasts.
  // For validation: store q in shared memory since D=128 is small.

  __shared__ __half q_shared[128];  // D max
  for (int i = tid; i < D; i += blockDim.x) {
    q_shared[i] = q_head[i];
  }
  __syncthreads();

  // --- Online softmax state ---
  // Each thread t (for t < D) tracks its assigned output dimension
  float o_acc = 0.0f;  // output[d] for d = tid
  float m_prev = -INFINITY;  // running max
  float l_prev = 0.0f;       // running sum

  const int num_tiles = L / TILE_L;

  for (int tile = 0; tile < num_tiles; ++tile) {
    // --- Load K tile into shared memory ---
    __shared__ __half KV_tile[TILE_L][128];  // [TILE_L][D] shared for K and V
    int l_start = tile * TILE_L;

    // Cooperative load of K tile: [TILE_L * D] elements, 256 threads
    for (int i = tid; i < TILE_L * D; i += blockDim.x) {
      int l = i / D;
      int d = i % D;
      KV_tile[l][d] = K_head[(l_start + l) * D + d];
    }
    __syncthreads();

    // --- Compute scores for this tile ---
    // Each thread computes one score value: S_tile[l] = dot(q, K_tile[l]) * scale
    // TILE_L=128, 256 threads → 2 score elements per thread
    __shared__ float S_tile[TILE_L];   // scores in shared
    __shared__ float P_tile[TILE_L];   // probs after local softmax

    for (int l = tid; l < TILE_L; l += blockDim.x) {
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) {
        dot += __half2float(q_shared[d]) * __half2float(KV_tile[l][d]);
      }
      S_tile[l] = dot * scale;
    }
    __syncthreads();

    // --- Tile-level softmax ---
    // Find max in this tile
    float m_tile = -INFINITY;
    for (int l = tid; l < TILE_L; l += blockDim.x) {
      if (S_tile[l] > m_tile) m_tile = S_tile[l];
    }

    // Warp reduce tile max
    __shared__ float warp_maxs[8];
    m_tile = device::block_reduce_max_broadcast(m_tile, warp_maxs, warp_id, lane, blockDim.x / 32);

    // Compute exp and local sum
    float l_tile = 0.0f;
    for (int l = tid; l < TILE_L; l += blockDim.x) {
      float p = expf(S_tile[l] - m_tile);
      P_tile[l] = p;
      l_tile += p;
    }

    __shared__ float warp_sums[8];
    l_tile = device::block_reduce_sum_broadcast(l_tile, warp_sums, warp_id, lane, blockDim.x / 32);
    __syncthreads();

    // --- Update running softmax ---
    float m_new = fmaxf(m_prev, m_tile);
    float exp_diff_prev = expf(m_prev - m_new);
    float exp_diff_tile = expf(m_tile - m_new);
    float l_new = exp_diff_prev * l_prev + exp_diff_tile * l_tile;

    // --- Accumulate weighted V ---
    // Load V tile into shared (reusing KV_tile)
    for (int i = tid; i < TILE_L * D; i += blockDim.x) {
      int l = i / D;
      int d = i % D;
      KV_tile[l][d] = V_head[(l_start + l) * D + d];
    }
    __syncthreads();

    // Update output: o_new[d] = (exp_diff_prev * l_prev * o_old[d] + exp_diff_tile * sum_l P_l * V_l[d]) / l_new
    // Each thread handles its assigned output dimension(s)
    if (tid < D) {
      float weighted_sum = 0.0f;
      for (int l = 0; l < TILE_L; ++l) {
        weighted_sum += P_tile[l] * __half2float(KV_tile[l][tid]);
      }
      o_acc = (exp_diff_prev * l_prev * o_acc + exp_diff_tile * weighted_sum) / l_new;
    }
    __syncthreads();

    m_prev = m_new;
    l_prev = l_new;
  }

  // --- Write output ---
  if (tid < D) {
    args.out[head * D + tid] = __float2half(o_acc);
  }
}

}  // namespace fused
}  // namespace kernels
}  // namespace decodebench_val
