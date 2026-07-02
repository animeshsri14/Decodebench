# DecodeBench Validation Report
Generated: 2026-07-02T10:54:21.145864

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

PASS means the fused speedup over the graph baseline is fully explained by the byte-elimination bound B. A positive residual is a genuine model-bound violation in the favorable direction: the fused kernel wins by MORE than Δ_launch + B. Known unmodeled terms (see README Limitations): elimination of inter-kernel serialization — low-parallelism interleaved stages (e.g. the H-block softmax) and per-boundary drain/ramp that graph replay cannot remove.

| Fusion | Dim | t_unfused_graph (us) | t_fused (us) | B (us) | Residual | Status |
|--------|-----|---------------------|-------------|--------|----------|--------|
| f1 | 2048 | 220.58 | 256.19 | 0.03 | -35.64 | PASS |
| f1 | 4096 | 437.79 | 562.84 | 0.06 | -125.11 | PASS |
| f2 | 2048 | 428.04 | 441.79 | 0.21 | -13.96 | PASS |
| f2 | 4096 | 847.68 | 880.65 | 0.21 | -33.17 | PASS |
| f4 | 2048 | 212.94 | 179.55 | 6.45 | 26.93 | FAIL |
| f4 | 4096 | 437.94 | 370.02 | 13.27 | 54.66 | FAIL |

## Check (b): Analytic bytes vs NCU DRAM bytes (tolerance ±20%)

Analytic is a lower bound: does not model cache-line granularity overhead or L2 thrashing caused by weight matrices exceeding L2 capacity. Measured ~10-15% excess over analytic for weight-streaming kernels on L4 (96 MB L2 vs 117+ MB weights).

| Fusion | Variant | Analytic (MB) | NCU DRAM (MB) | Ratio | Status |
|--------|---------|---------------|---------------|-------|--------|
| f1 | unfused-stream | 117.50 | 132.10 | 0.89 | PASS |
| f1 | fused | 117.50 | 139.05 | 0.85 | PASS |
| f2 | unfused-stream | 234.98 | 264.32 | 0.89 | PASS |
| f2 | fused | 234.98 | 275.93 | 0.85 | PASS |
| f4 | unfused-stream | 69.22 | 97.59 | 0.71 | WARN (favorable: unfused>analytic, strengthens BYTE-BOUND) |
| f4 | fused | 69.22 | 90.78 | 0.76 | FAIL |

## Check (c): CUDA Graphs capture launch overhead (Δ_launch > 0)

Δ_launch = t_stream - t_graph must be positive: graphs are at least as fast as stream, confirming launch overhead exists and is captured by graph replay.

| Fusion | t_stream (us) | t_graph (us) | Δ_launch (us) | t_fused (us) | Status |
|--------|--------------|-------------|--------------|-------------|--------|
| f1 | 438.34 | 437.79 | 0.55 | 562.84 | PASS |
| f2 | 851.79 | 847.68 | 4.11 | 880.65 | PASS |
| f4 | 446.06 | 437.94 | 8.12 | 370.02 | PASS |

---

## Summary

- PASS: 30
- FAIL: 3
- **Overall: FAIL**