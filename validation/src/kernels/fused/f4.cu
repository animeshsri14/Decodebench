// f4.cu — FlashDecode-style fused attention kernels
//
// Two implementations:
//   1. f4_kernel — single block per head, correctness reference. Simple but
//      leaves most SMs idle (grid = H); kept as the documented baseline.
//   2. f4_partial_kernel + f4_reduce_kernel — throughput-tuned split-KV
//      FlashDecode. Each head's KV range is divided across n_splits blocks
//      (grid = H * n_splits) that stream K/V with vectorized float4 loads and
//      keep an online softmax over their chunk; a second kernel merges the
//      per-split (m, l, o) partials via log-sum-exp. This is the "fused"
//      variant benchmarked by bench_variant: 2 launches, scores/probs never
//      written to global memory.
//
// Both require D <= 128 and L a multiple of TILE_L.

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

// =====================================================================
// Split-KV FlashDecode (throughput-tuned)
// =====================================================================

// Phase 1: each block owns (head, split) and runs an online softmax over its
// contiguous chunk of KV tiles, writing UNNORMALIZED partials:
//   part_m[h,s] = max score seen, part_l[h,s] = sum exp(score - m),
//   part_o[h,s,d] = sum exp(score - m) * V[d]
// grid = H * n_splits, block = 256. Empty splits (when n_splits does not
// divide the tile count evenly) write m=-inf, l=0, o=0 and drop out of the
// merge naturally.
//
// K and V are read directly from global with the SAME access idioms as the
// tuned unfused kernels (warp-per-row K dots; consecutive-lane V dims), so
// the fused-vs-unfused comparison isolates eliminated intermediate bytes and
// launches rather than kernel tuning quality. Only scores/probs live in
// shared memory — they never touch global.
__global__ void f4_partial_kernel(KernelArgs args) {
  const int H     = args.H;
  const int L     = args.L;
  const int D     = args.D;
  const int S     = args.n_splits;
  const int head  = blockIdx.x / S;
  const int split = blockIdx.x % S;
  const int tid   = threadIdx.x;
  const int lane    = tid % 32;
  const int warp_id = tid / 32;

  if (head >= H) return;

  const int num_tiles       = L / TILE_L;
  const int tiles_per_split = (num_tiles + S - 1) / S;
  const int tile_begin      = split * tiles_per_split;
  const int tile_end        = min(num_tiles, tile_begin + tiles_per_split);

  const __half* q_head = args.q + head * D;
  const __half* K_head = args.K + static_cast<size_t>(head) * L * D;
  const __half* V_head = args.V + static_cast<size_t>(head) * L * D;

  const float scale = rsqrtf(static_cast<float>(D));

  __shared__ __half q_shared[128];
  for (int i = tid; i < D; i += blockDim.x) {
    q_shared[i] = q_head[i];
  }

  __shared__ float S_tile[TILE_L];
  __shared__ float P_tile[TILE_L];
  __shared__ float pv_hi[128];
  __shared__ float warp_maxs[8];
  __shared__ float warp_sums[8];
  __syncthreads();

  float o_acc  = 0.0f;       // unnormalized output accumulator for d = tid%D
  float m_prev = -INFINITY;
  float l_prev = 0.0f;

  for (int tile = tile_begin; tile < tile_end; ++tile) {
    const int l_start = tile * TILE_L;

    // --- Scores: warp-cooperative row dots straight from global K.
    //     Lanes read consecutive halves — coalesced 64B segments, the same
    //     idiom as the unfused attn_scores kernel. ---
    const __half2* qrow = reinterpret_cast<const __half2*>(q_shared);
    for (int l = warp_id; l < TILE_L; l += blockDim.x / 32) {
      const __half2* krow = reinterpret_cast<const __half2*>(
          K_head + static_cast<size_t>(l_start + l) * D);
      float dot = 0.0f;
      for (int i = lane; i < D / 2; i += 32) {
        const float2 kf = __half22float2(krow[i]);
        const float2 qf = __half22float2(qrow[i]);
        dot += kf.x * qf.x + kf.y * qf.y;
      }
      for (int off = 16; off > 0; off >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, off);
      }
      if (lane == 0) S_tile[l] = dot * scale;
    }
    __syncthreads();

    // --- Tile max and exp-sum ---
    float m_tile = -INFINITY;
    for (int l = tid; l < TILE_L; l += blockDim.x) {
      m_tile = fmaxf(m_tile, S_tile[l]);
    }
    m_tile = device::block_reduce_max_broadcast(m_tile, warp_maxs, warp_id, lane, blockDim.x / 32);

    float l_tile = 0.0f;
    for (int l = tid; l < TILE_L; l += blockDim.x) {
      float p = expf(S_tile[l] - m_tile);
      P_tile[l] = p;
      l_tile += p;
    }
    l_tile = device::block_reduce_sum_broadcast(l_tile, warp_sums, warp_id, lane, blockDim.x / 32);
    __syncthreads();

    // --- P·V straight from global V with all 256 threads: lower half sums
    //     rows [0, TILE_L/2), upper half rows [TILE_L/2, TILE_L) for the
    //     same output dim. Consecutive lanes touch consecutive dims —
    //     coalesced, the same idiom as the unfused attn_v kernel. ---
    const float m_new         = fmaxf(m_prev, m_tile);
    const float exp_diff_prev = expf(m_prev - m_new);
    const float exp_diff_tile = expf(m_tile - m_new);

    const int d      = tid & 127;
    const int row_lo = (tid < 128) ? 0 : TILE_L / 2;
    float wsum = 0.0f;
    if (d < D) {
      for (int l = row_lo; l < row_lo + TILE_L / 2; ++l) {
        wsum += P_tile[l] *
                __half2float(V_head[static_cast<size_t>(l_start + l) * D + d]);
      }
    }
    if (tid >= 128 && d < D) pv_hi[d] = wsum;
    __syncthreads();

    if (tid < D) {
      const float tile_sum = wsum + pv_hi[d];
      o_acc  = o_acc * exp_diff_prev + tile_sum * exp_diff_tile;
    }
    m_prev = m_new;
    l_prev = l_prev * exp_diff_prev + l_tile * exp_diff_tile;
    __syncthreads();
  }

  // --- Write partials ---
  const int part_idx = head * S + split;
  if (tid == 0) {
    args.part_m[part_idx] = m_prev;
    args.part_l[part_idx] = l_prev;
  }
  if (tid < D) {
    args.part_o[static_cast<size_t>(part_idx) * D + tid] = o_acc;
  }
}

// Phase 2: merge the n_splits partials per head via log-sum-exp and write the
// normalized FP16 output. grid = H, block = D.
__global__ void f4_reduce_kernel(KernelArgs args) {
  const int H    = args.H;
  const int D    = args.D;
  const int S    = args.n_splits;
  const int head = blockIdx.x;
  const int tid  = threadIdx.x;

  if (head >= H || tid >= D) return;

  const float* pm = args.part_m + head * S;
  const float* pl = args.part_l + head * S;
  const float* po = args.part_o + static_cast<size_t>(head) * S * D;

  float m_max = -INFINITY;
  for (int s = 0; s < S; ++s) m_max = fmaxf(m_max, pm[s]);

  float l_tot = 0.0f;
  float o_tot = 0.0f;
  for (int s = 0; s < S; ++s) {
    const float w = expf(pm[s] - m_max);
    l_tot += w * pl[s];
    o_tot += w * po[s * D + tid];
  }

  args.out[head * D + tid] = __float2half(o_tot / l_tot);
}

}  // namespace fused
}  // namespace kernels
}  // namespace decodebench_val
