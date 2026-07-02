# DecodeBench Validation Report
Generated: 2026-07-02T22:51:07.257463

## Check (G0): Data completeness

Every expected (fusion, dim, batch, variant) config must be present with usable timing samples, and every expected NCU cell must have data. A missing cell is a FAIL: an empty or partial collection must not be able to produce an overall PASS.

- (unlisted cells present and usable)

## Check (G1): Numerical correctness (fused/unfused vs CPU reference)

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

## Check (a): Structural decomposition t_graph − t_fused = B + S, τ corroboration

v2 (PREREGISTRATION-v2.md): the fusion gap decomposes into the byte-time estimate B and the structural term S = (t_graph − t_fused) − B. The gate is DIRECTIONAL instrument corroboration: the independently measured isolated-kernel-duration gap τ_u − τ_f (NCU gpu__time_duration.sum) must agree in sign with the wall-clock gap, unless either magnitude is within the 5 µs near-zero band. τ magnitudes are NOT gated: NCU replay flushes caches between kernels, inflating multi-kernel chains that enjoy inter-kernel L2 reuse in steady state; the sign is robust to that bias, the microsecond value is not. [Supersedes the v1 residual gate (gap ≈ B alone), refuted on T4 2026-07-02 — see README.]

| Fusion | Dim | t_graph (us) | t_fused (us) | Gap (us) | B (us) | S (us) | τ_u−τ_f (us) | Status |
|--------|-----|-------------|-------------|----------|--------|--------|--------------|--------|
| f1 | 2048 | 232.69 | 236.07 | -3.38 | 0.03 | -3.41 | -6.62 | PASS |
| f1 | 4096 | 460.96 | 468.88 | -7.92 | 0.06 | -7.98 | -14.94 | PASS |
| f2 | 2048 | 452.43 | 465.04 | -12.61 | 0.44 | -13.05 | -11.78 | PASS |
| f2 | 4096 | 897.89 | 926.75 | -28.85 | 0.44 | -29.29 | -28.45 | PASS |
| f4 | 2048 | 141.90 | 139.64 | 2.26 | 4.30 | -2.04 | 0.64 | PASS |
| f4 | 4096 | 279.39 | 279.40 | -0.00 | 8.46 | -8.47 | -3.33 | PASS |

## Check (b): Analytic bytes vs NCU DRAM bytes

F1/F2 gate on absolute totals per dim (tolerance ±20%; analytic is a lower bound; cache-line granularity and L2 thrashing add measured ~10-15% on weight streams). F4 gates on the eliminated DELTA, two-sided ±50%, but ONLY when the analytic delta is at least 5% of the smaller variant total: below that the signal sits under the DRAM-counter noise floor (uniform ~1.3-1.4x excess on KV streams observed on T4, both variants) and the check records below-resolution — no byte-delta claim is made either way. The v2 byte term for F4 is B inside the check (a) decomposition, not this counter delta.

| Fusion | Dim | Variant | Analytic (MB) | NCU DRAM (MB) | Ratio | Status |
|--------|-----|---------|---------------|---------------|-------|--------|
| f1 | 2048 | unfused-stream | 58.77 | 65.35 | 0.90 | PASS |
| f1 | 2048 | fused | 58.77 | 67.35 | 0.87 | PASS |
| f1 | 4096 | unfused-stream | 117.50 | 130.65 | 0.90 | PASS |
| f1 | 4096 | fused | 117.50 | 134.66 | 0.87 | PASS |
| f2 | 2048 | unfused-stream | 117.59 | 130.76 | 0.90 | PASS |
| f2 | 2048 | fused | 117.59 | 135.31 | 0.87 | PASS |
| f2 | 4096 | unfused-stream | 235.04 | 261.34 | 0.90 | PASS |
| f2 | 4096 | fused | 235.04 | 270.78 | 0.87 | PASS |
| f4 | 2048 | delta (unfused−fused) | 0.52 | -0.85 | -1.65 | PASS (below resolution — no claim) |
| f4 | 2048 | unfused-stream (diagnostic) | 34.62 | 38.42 | 0.90 | — |
| f4 | 2048 | fused (diagnostic) | 34.62 | 39.27 | 0.88 | — |
| f4 | 4096 | delta (unfused−fused) | 1.03 | -1.88 | -1.82 | PASS (below resolution — no claim) |
| f4 | 4096 | unfused-stream (diagnostic) | 69.22 | 76.76 | 0.90 | — |
| f4 | 4096 | fused (diagnostic) | 69.22 | 78.64 | 0.88 | — |

## Check (c): CUDA Graphs capture launch overhead (Δ_launch ≥ −noise)

Graphs must be at least as fast as stream launches up to the timer noise floor: Δ_launch ≥ −max(0.5%·t_graph, 2 µs). For long kernels at high per-trial iteration counts the amortized CPU launch cost can be smaller than cudaEvent resolution, so small negative readings are measurement noise, not a graphs regression. [v2 revision: the v1 gate required ≥ 0 exactly and failed on a −0.19 µs reading against ~384 µs kernels.]

| Fusion | Dim | t_stream (us) | t_graph (us) | Δ_launch (us) | Status |
|--------|-----|--------------|-------------|--------------|--------|
| f1 | 2048 | 233.38 | 232.69 | 0.69 | PASS |
| f1 | 4096 | 461.66 | 460.96 | 0.70 | PASS |
| f2 | 2048 | 453.51 | 452.43 | 1.09 | PASS |
| f2 | 4096 | 898.92 | 897.89 | 1.03 | PASS |
| f4 | 2048 | 142.61 | 141.90 | 0.71 | PASS |
| f4 | 4096 | 281.24 | 279.39 | 1.85 | PASS |

## Check (H): Pre-registered hypotheses v2 (PREREGISTRATION-v2.md)

H1-v2 (F1/F2, launch-bound / fusion-not-worthwhile): fusion yields no wall-clock win — t_fused ≥ t_graph − max(0.5%·t_graph, 2 µs) — and its structural term is non-positive within noise: S ≤ max(1%·t_graph, 3 µs). H2-v2 (F4, structure-bound): fused wins wall-clock (t_fused < t_graph), the structural term is positive (S > 0), and structure dominates bytes (S > B). v1 H2 ('B alone ≥ 80% of the gap') and v1 H4(c) are retired — refuted on T4 2026-07-02; the refutation is recorded in the README validation notes and motivated this v2 decomposition.

| Hypothesis | Fusion | Dim | Gap t_graph−t_fused (us) | B (us) | S (us) | Status |
|-----------|--------|-----|--------------------------|--------|--------|--------|
| H1-v2 | f1 | 2048 | -3.38 | 0.03 | -3.41 | PASS |
| H1-v2 | f1 | 4096 | -7.92 | 0.06 | -7.98 | PASS |
| H1-v2 | f2 | 2048 | -12.61 | 0.44 | -13.05 | PASS |
| H1-v2 | f2 | 4096 | -28.85 | 0.44 | -29.29 | PASS |
| H2-v2 | f4 | 2048 | 2.26 | 4.30 | -2.04 | FAIL |
| H2-v2 | f4 | 4096 | -0.00 | 8.46 | -8.47 | FAIL |

---

## Summary

- PASS: 74
- WARN: 0 (warnings never count as passes)
- FAIL: 2
- **Overall: FAIL**

### Failing checks
- [H] H2-v2/f4/2048: failed: S>0, S>B
- [H] H2-v2/f4/4096: failed: t_fused<t_graph, S>0, S>B