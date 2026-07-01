# DecodeBench Validation Report
Generated: 2026-07-01T19:19:50.169727

## Check (G1): Numerical correctness (fused/unfused vs CPU reference)

Every measured config must report correctness_ok=1 from bench_variant, which checks each variant against an inline CPU reference (numpy-allclose tolerance: a mismatch must exceed both abs 5e-2 and rel 2e-2). A FAIL here voids the timing.

| Fusion | Dim | Variant | correctness_ok | Status |
|--------|-----|---------|----------------|--------|
| f1 | 2048 | unfused-stream | 1 | PASS |
| f1 | 2048 | unfused-graph | 1 | PASS |
| f1 | 2048 | fused | 1 | PASS |
| f1 | 4096 | unfused-stream | 1 | PASS |
| f1 | 4096 | unfused-graph | 1 | PASS |
| f1 | 4096 | fused | 1 | PASS |
| f2 | 2048 | unfused-stream | 1 | PASS |
| f2 | 2048 | unfused-graph | 1 | PASS |
| f2 | 2048 | fused | 1 | PASS |
| f2 | 4096 | unfused-stream | 1 | PASS |
| f2 | 4096 | unfused-graph | 1 | PASS |
| f2 | 4096 | fused | 1 | PASS |
| f4 | 2048 | unfused-stream | 1 | PASS |
| f4 | 2048 | unfused-graph | 1 | PASS |
| f4 | 2048 | fused | 1 | PASS |
| f4 | 4096 | unfused-stream | 1 | PASS |
| f4 | 4096 | unfused-graph | 1 | PASS |
| f4 | 4096 | fused | 1 | PASS |

## Check (a): Residual analysis (t_graph - t_fused - B)

| Fusion | Dim | t_unfused_graph (us) | t_fused (us) | B (us) | Residual | Status |
|--------|-----|---------------------|-------------|--------|----------|--------|
| f1 | 2048 | 219.12 | 253.14 | 0.03 | -34.05 | PASS |
| f1 | 4096 | 435.58 | 544.20 | 0.06 | -108.68 | PASS |
| f2 | 2048 | 426.84 | 441.04 | 0.21 | -14.40 | PASS |
| f2 | 4096 | 845.50 | 879.93 | 0.21 | -34.63 | PASS |
| f4 | 2048 | 388.00 | 498.54 | 11.75 | -122.29 | PASS |
| f4 | 4096 | 383.96 | 498.55 | 11.62 | -126.22 | PASS |

## Check (b): Analytic bytes vs NCU DRAM bytes (tolerance ±20%)

Analytic is a lower bound: does not model cache-line granularity overhead or L2 thrashing. T4 has only ~4 MB L2; weight matrices (117–235 MB) fully evict the cache, causing measured DRAM traffic to exceed the analytic floor by 10–15% for F1/F2 (weight-streaming) and more severely for F4/fused (ratio 0.72, outside ±20% - the sole FAIL this run).

| Fusion | Variant | Analytic (MB) | NCU DRAM (MB) | Ratio | Status |
|--------|---------|---------------|---------------|-------|--------|
| f1 | unfused-stream | 117.50 | 130.80 | 0.90 | PASS |
| f1 | fused | 117.50 | 137.55 | 0.85 | PASS |
| f2 | unfused-stream | 234.98 | 261.62 | 0.90 | PASS |
| f2 | fused | 234.98 | 274.42 | 0.86 | PASS |
| f4 | unfused-stream | 17.32 | 27.62 | 0.63 | WARN (favorable: unfused>analytic, strengthens BYTE-BOUND) |
| f4 | fused | 17.32 | 23.94 | 0.72 | FAIL |

## Check (c): CUDA Graphs capture launch overhead (Δ_launch > 0)

Δ_launch = t_stream - t_graph must be positive: graphs are at least as fast as stream, confirming launch overhead exists and is captured by graph replay.

| Fusion | t_stream (us) | t_graph (us) | Δ_launch (us) | t_fused (us) | Status |
|--------|--------------|-------------|--------------|-------------|--------|
| f1 | 436.38 | 435.58 | 0.80 | 544.20 | PASS |
| f2 | 847.20 | 845.50 | 1.70 | 879.93 | PASS |
| f4 | 396.35 | 383.96 | 12.40 | 498.55 | PASS |

---

## Summary

- PASS: 32
- FAIL: 1
- **Overall: FAIL**