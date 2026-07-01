#pragma once
// warp_reduce.h — shared warp reduction primitives for DecodeBench validation kernels
// All reductions use __shfl_xor_sync with the active warp mask, no shared-memory atomics.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace decodebench_val {
namespace device {

// ---- float (FP32) reductions ----

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    float other = __shfl_xor_sync(0xffffffff, val, mask);
    val = (val > other) ? val : other;
  }
  return val;
}

// ---- half (FP16) reduction to FP32 ----
// Accumulate in FP32, broadcast result as FP32; caller casts back if needed.

__device__ __forceinline__ float warp_reduce_sum_half(__half val) {
  float acc = __half2float(val);
  return warp_reduce_sum(acc);
}

// ---- half2 reductions ----

__device__ __forceinline__ float2 warp_reduce_sum_half2(__half2 val) {
  float2 acc;
  acc.x = __half2float(__low2half(val));
  acc.y = __half2float(__high2half(val));
  for (int mask = 16; mask > 0; mask >>= 1) {
    float2 other;
    __half2 other_h2 = __shfl_xor_sync(0xffffffff, val, mask);
    other.x = __half2float(__low2half(other_h2));
    other.y = __half2float(__high2half(other_h2));
    acc.x += other.x;
    acc.y += other.y;
  }
  return acc;
}

// ---- block-wide reductions ----
// Each warp reduces, then lane 0 of each warp writes to shared memory.
// Caller provides shared memory of at least warp_count * sizeof(float).

__device__ __forceinline__ float block_reduce_sum(float val,
                                                   float* smem,
                                                   int warp_id,
                                                   int lane_id,
                                                   int warp_count) {
  val = warp_reduce_sum(val);
  if (lane_id == 0) {
    smem[warp_id] = val;
  }
  __syncthreads();
  // Warp 0 reads the per-warp partials and reduces. Only smem[0..warp_count)
  // are valid — lanes past warp_count must read 0, not whatever is sitting in
  // shared memory beyond the array, or the sum picks up garbage. (Same guard
  // as the _broadcast variants below.)
  float result = (warp_id == 0 && lane_id < warp_count) ? smem[lane_id] : 0.0f;
  if (warp_id == 0) {
    result = warp_reduce_sum(result);
  }
  return result; // only valid on warp 0, lane 0; broadcast to all if needed
}

// Block-wide max broadcast
__device__ __forceinline__ float block_reduce_max_broadcast(float val,
                                                             float* smem,
                                                             int warp_id,
                                                             int lane_id,
                                                             int warp_count) {
  val = warp_reduce_max(val);
  if (lane_id == 0) {
    smem[warp_id] = val;
  }
  __syncthreads();
  // Guard with warp_id==0 so all 32 threads of warp 0 participate in
  // warp_reduce_max. On Volta+ (SM70+) independent thread scheduling,
  // __shfl_xor_sync with full mask deadlocks if only a subset of warp
  // lanes executes it (threads 0-7 wait forever for threads 8-31).
  if (warp_id == 0) {
    val = (lane_id < warp_count) ? smem[lane_id] : -INFINITY;
    val = warp_reduce_max(val);
    if (lane_id == 0) smem[0] = val;
  }
  __syncthreads();
  return smem[0];
}

// Full block broadcast: all threads get the same reduced value.
__device__ __forceinline__ float block_reduce_sum_broadcast(float val,
                                                             float* smem,
                                                             int warp_id,
                                                             int lane_id,
                                                             int warp_count) {
  val = warp_reduce_sum(val);
  if (lane_id == 0) {
    smem[warp_id] = val;
  }
  __syncthreads();
  // Guard with warp_id==0 (same Volta+ deadlock fix as block_reduce_max_broadcast).
  if (warp_id == 0) {
    val = (lane_id < warp_count) ? smem[lane_id] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane_id == 0) smem[0] = val;
  }
  __syncthreads();
  return smem[0];
}

// ---- vectorized load helpers ----

// Load 8 FP16 values (uint4 = 16 bytes) from a global pointer.
// Returns as 4 half2 values packed in a struct for convenience.
struct Vec8Half {
  __half2 v0, v1, v2, v3;
};

__device__ __forceinline__ Vec8Half load_vec8(const __half* ptr) {
  Vec8Half out;
  // Reinterpret as uint4 for 128-bit load
  const uint4* u4 = reinterpret_cast<const uint4*>(ptr);
  uint4 data = *u4;

  // Unpack uint4 → 8 half values → 4 half2
  // uint4 layout: x, y, z, w each hold 2 halves (4 bytes each)
  auto unpack2 = [](unsigned int word) -> __half2 {
    return __halves2half2(
        __ushort_as_half(static_cast<unsigned short>(word & 0xFFFF)),
        __ushort_as_half(static_cast<unsigned short>((word >> 16) & 0xFFFF)));
  };
  out.v0 = unpack2(data.x);
  out.v1 = unpack2(data.y);
  out.v2 = unpack2(data.z);
  out.v3 = unpack2(data.w);

  return out;
}

}  // namespace device
}  // namespace decodebench_val
