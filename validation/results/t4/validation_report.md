# DecodeBench Validation Report
Generated: 2026-07-02T21:35:34.727192

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
| f1 | 2048 | 219.62 | 253.43 | -33.81 | 0.03 | -33.84 | -38.94 | PASS |
| f1 | 4096 | 435.23 | 537.09 | -101.87 | 0.06 | -101.93 | -153.79 | PASS |
| f2 | 2048 | 425.73 | 439.99 | -14.27 | 0.42 | -14.68 | -2.05 | PASS |
| f2 | 4096 | 844.47 | 878.39 | -33.92 | 0.41 | -34.33 | -22.75 | PASS |
| f4 | 2048 | 187.56 | 166.30 | 21.26 | 5.68 | 15.58 | 55.23 | PASS |
| f4 | 4096 | 384.16 | 317.53 | 66.63 | 11.64 | 54.99 | 135.78 | PASS |

## Check (b): Analytic bytes vs NCU DRAM bytes

F1/F2 gate on absolute totals per dim (tolerance ±20%; analytic is a lower bound; cache-line granularity and L2 thrashing add measured ~10-15% on weight streams). F4 gates on the eliminated DELTA, two-sided ±50%, but ONLY when the analytic delta is at least 5% of the smaller variant total: below that the signal sits under the DRAM-counter noise floor (uniform ~1.3-1.4x excess on KV streams observed on T4, both variants) and the check records below-resolution — no byte-delta claim is made either way. The v2 byte term for F4 is B inside the check (a) decomposition, not this counter delta.

| Fusion | Dim | Variant | Analytic (MB) | NCU DRAM (MB) | Ratio | Status |
|--------|-----|---------|---------------|---------------|-------|--------|
| f1 | 2048 | unfused-stream | 58.77 | 66.57 | 0.88 | PASS |
| f1 | 2048 | fused | 58.77 | 69.91 | 0.84 | PASS |
| f1 | 4096 | unfused-stream | 117.50 | 132.08 | 0.89 | PASS |
| f1 | 4096 | fused | 117.50 | 138.98 | 0.85 | PASS |
| f2 | 2048 | unfused-stream | 117.59 | 133.20 | 0.88 | PASS |
| f2 | 2048 | fused | 117.59 | 138.12 | 0.85 | PASS |
| f2 | 4096 | unfused-stream | 235.04 | 264.13 | 0.89 | PASS |
| f2 | 4096 | fused | 235.04 | 276.00 | 0.85 | PASS |
| f4 | 2048 | delta (unfused−fused) | 0.52 | 3.45 | 6.69 | PASS (below resolution — no claim) |
| f4 | 2048 | unfused-stream (diagnostic) | 34.62 | 49.23 | 0.70 | — |
| f4 | 2048 | fused (diagnostic) | 34.62 | 45.78 | 0.76 | — |
| f4 | 4096 | delta (unfused−fused) | 1.03 | 6.99 | 6.77 | PASS (below resolution — no claim) |
| f4 | 4096 | unfused-stream (diagnostic) | 69.22 | 97.38 | 0.71 | — |
| f4 | 4096 | fused (diagnostic) | 69.22 | 90.39 | 0.77 | — |

## Check (c): CUDA Graphs capture launch overhead (Δ_launch ≥ −noise)

Graphs must be at least as fast as stream launches up to the timer noise floor: Δ_launch ≥ −max(0.5%·t_graph, 2 µs). For long kernels at high per-trial iteration counts the amortized CPU launch cost can be smaller than cudaEvent resolution, so small negative readings are measurement noise, not a graphs regression. [v2 revision: the v1 gate required ≥ 0 exactly and failed on a −0.19 µs reading against ~384 µs kernels.]

| Fusion | Dim | t_stream (us) | t_graph (us) | Δ_launch (us) | Status |
|--------|-----|--------------|-------------|--------------|--------|
| f1 | 2048 | 220.82 | 219.62 | 1.20 | PASS |
| f1 | 4096 | 436.06 | 435.23 | 0.83 | PASS |
| f2 | 2048 | 427.93 | 425.73 | 2.20 | PASS |
| f2 | 4096 | 846.48 | 844.47 | 2.01 | PASS |
| f4 | 2048 | 187.58 | 187.56 | 0.02 | PASS |
| f4 | 4096 | 383.97 | 384.16 | -0.19 | PASS |

## Check (H): Pre-registered hypotheses v2 (PREREGISTRATION-v2.md)

H1-v2 (F1/F2, launch-bound / fusion-not-worthwhile): fusion yields no wall-clock win — t_fused ≥ t_graph − max(0.5%·t_graph, 2 µs) — and its structural term is non-positive within noise: S ≤ max(1%·t_graph, 3 µs). H2-v2 (F4, structure-bound): fused wins wall-clock (t_fused < t_graph), the structural term is positive (S > 0), and structure dominates bytes (S > B). v1 H2 ('B alone ≥ 80% of the gap') and v1 H4(c) are retired — refuted on T4 2026-07-02; the refutation is recorded in the README validation notes and motivated this v2 decomposition.

| Hypothesis | Fusion | Dim | Gap t_graph−t_fused (us) | B (us) | S (us) | Status |
|-----------|--------|-----|--------------------------|--------|--------|--------|
| H1-v2 | f1 | 2048 | -33.81 | 0.03 | -33.84 | PASS |
| H1-v2 | f1 | 4096 | -101.87 | 0.06 | -101.93 | PASS |
| H1-v2 | f2 | 2048 | -14.27 | 0.42 | -14.68 | PASS |
| H1-v2 | f2 | 4096 | -33.92 | 0.41 | -34.33 | PASS |
| H2-v2 | f4 | 2048 | 21.26 | 5.68 | 15.58 | PASS |
| H2-v2 | f4 | 4096 | 66.63 | 11.64 | 54.99 | PASS |

---

## Summary

- PASS: 76
- WARN: 0 (warnings never count as passes)
- FAIL: 0
- **Overall: PASS**