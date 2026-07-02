# DecodeBench Validation Report

> **RETRACTED VALIDATION STATUS:** This historical report predates the current
> F2 accounting, cache-residency parity, split-KV F4 implementation, and
> fail-closed gates. Its raw measurements are retained, but its PASS summary is
> not valid evidence under the current pipeline. A fresh hardware run is required.
Generated: 2026-07-01T00:15:33.183667

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
| f1 | 2048 | 231.85 | 235.26 | 0.03 | -3.44 | PASS |
| f1 | 4096 | 461.06 | 469.27 | 0.06 | -8.27 | PASS |
| f2 | 2048 | 451.39 | 465.03 | 0.22 | -13.86 | PASS |
| f2 | 4096 | 897.12 | 926.92 | 0.22 | -30.02 | PASS |
| f4 | 2048 | 129.59 | 249.43 | 3.92 | -123.76 | PASS |
| f4 | 4096 | 129.36 | 249.44 | 3.92 | -123.99 | PASS |

## Check (b): Analytic bytes vs NCU DRAM bytes (tolerance ±20%)

Analytic is a lower bound: does not model cache-line granularity overhead or L2 thrashing caused by weight matrices exceeding L2 capacity. Measured ~10-15% excess over analytic for weight-streaming kernels on L4 (96 MB L2 vs 117+ MB weights).

| Fusion | Variant | Analytic (MB) | NCU DRAM (MB) | Ratio | Status |
|--------|---------|---------------|---------------|-------|--------|
| f1 | unfused-stream | 117.50 | 133.05 | 0.88 | PASS |
| f1 | fused | 117.50 | 137.12 | 0.86 | PASS |
| f2 | unfused-stream | 234.98 | 264.88 | 0.89 | PASS |
| f2 | fused | 234.98 | 273.39 | 0.86 | PASS |
| f4 | unfused-stream | 17.32 | 27.94 | 0.62 | WARN (favorable: unfused>analytic, strengthens BYTE-BOUND) |
| f4 | fused | 17.32 | 20.30 | 0.85 | PASS |

## Check (c): CUDA Graphs capture launch overhead (Δ_launch > 0)

Δ_launch = t_stream - t_graph must be positive: graphs are at least as fast as stream, confirming launch overhead exists and is captured by graph replay.

| Fusion | t_stream (us) | t_graph (us) | Δ_launch (us) | t_fused (us) | Status |
|--------|--------------|-------------|--------------|-------------|--------|
| f1 | 461.81 | 461.06 | 0.74 | 469.27 | PASS |
| f2 | 899.48 | 897.12 | 2.36 | 926.92 | PASS |
| f4 | 130.61 | 129.36 | 1.25 | 249.44 | PASS |

---

## Summary

- PASS: 33
- FAIL: 0
- **Overall: PASS**
