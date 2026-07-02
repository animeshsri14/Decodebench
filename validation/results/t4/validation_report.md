# DecodeBench Validation Report
Generated: 2026-07-02T18:12:56.222296

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

## Check (a): Residual analysis (t_graph - t_fused - B)

PASS means the fused speedup over the graph baseline is explained by the byte estimate B within +2% of t_graph. A larger residual — in either the unfavorable OR the favorable direction — is a FAIL: a fused win far beyond Δ_launch + B means the decomposition does not describe this workload, whatever the sign of the surprise. [2026-07 revision: the earlier favorable-direction WARN reclassification is reverted.]

| Fusion | Dim | t_graph (us) | t_fused (us) | B (us) | Residual | Status |
|--------|-----|-------------|-------------|--------|----------|--------|
| f1 | 2048 | 219.62 | 253.43 | 0.03 | -33.84 | FAIL |
| f1 | 4096 | 435.23 | 537.09 | 0.06 | -101.93 | FAIL |
| f2 | 2048 | 425.73 | 439.99 | 0.42 | -14.68 | FAIL |
| f2 | 4096 | 844.47 | 878.39 | 0.41 | -34.33 | FAIL |
| f4 | 2048 | 187.56 | 166.30 | 5.68 | 15.58 | FAIL |
| f4 | 4096 | 384.16 | 317.53 | 11.64 | 54.99 | FAIL |

## Check (b): Analytic bytes vs NCU DRAM bytes

F1/F2 gate on absolute totals (tolerance ±20%; analytic is a lower bound; cache-line granularity and L2 thrashing add measured ~10-15% on weight streams). F4 gates on the eliminated DELTA, two-sided: measured (unfused − fused) DRAM bytes must lie within ±50% of the analytic delta (eliminable bytes minus the modeled split-KV partial-buffer traffic the fused variant adds). A delta far ABOVE the model is as much a model failure as one below it — an unbounded one-sided gate would pass on any unrelated traffic difference.

| Fusion | Variant | Analytic (MB) | NCU DRAM (MB) | Ratio | Status |
|--------|---------|---------------|---------------|-------|--------|
| f1 | unfused-stream | 117.50 | 132.05 | 0.89 | PASS |
| f1 | fused | 117.50 | 138.96 | 0.85 | PASS |
| f2 | unfused-stream | 235.04 | 264.18 | 0.89 | PASS |
| f2 | fused | 235.04 | 275.98 | 0.85 | PASS |
| f4 | delta (unfused−fused) | 1.03 | 6.52 | 6.32 | FAIL |
| f4 | unfused-stream (diagnostic) | 69.22 | 97.05 | 0.71 | — |
| f4 | fused (diagnostic) | 69.22 | 90.52 | 0.76 | — |

## Check (c): CUDA Graphs capture launch overhead (Δ_launch ≥ 0)

| Fusion | Dim | t_stream (us) | t_graph (us) | Δ_launch (us) | Status |
|--------|-----|--------------|-------------|--------------|--------|
| f1 | 2048 | 220.82 | 219.62 | 1.20 | PASS |
| f1 | 4096 | 436.06 | 435.23 | 0.83 | PASS |
| f2 | 2048 | 427.93 | 425.73 | 2.20 | PASS |
| f2 | 4096 | 846.48 | 844.47 | 2.01 | PASS |
| f4 | 2048 | 187.58 | 187.56 | 0.02 | PASS |
| f4 | 4096 | 383.97 | 384.16 | -0.19 | FAIL |

## Check (H): Pre-registered hypotheses (README)

H1: F1/F2 launch-bound — Δ_launch explains ≥80% of the unfused-to-fused gap (t_stream − t_fused). H2: F4 byte-bound — B explains ≥80% of that gap. H4(c): |(t_stream − t_fused) − (Δ_launch + B)| ≤ 2% of t_graph. These are gates, not notes: a refuted hypothesis is a FAIL in this report (and belongs in the paper as a negative result). When the fused kernel provides no gain over the stream baseline (gap ≤ 0) the attribution fraction is undefined; that is reported as WARN with the gap shown, since the pre-registered claim is about a positive gap.

| Hypothesis | Fusion | Dim | Gap (us) | Term (us) | Fraction | Status |
|-----------|--------|-----|----------|-----------|----------|--------|
| H1 | f1 | 2048 | -32.61 | 1.20 | — | WARN |
| H4(c) | f1 | 2048 | -32.61 | 1.23 | — | FAIL |
| H1 | f1 | 4096 | -101.03 | 0.83 | — | WARN |
| H4(c) | f1 | 4096 | -101.03 | 0.90 | — | FAIL |
| H1 | f2 | 2048 | -12.06 | 2.20 | — | WARN |
| H4(c) | f2 | 2048 | -12.06 | 2.62 | — | FAIL |
| H1 | f2 | 4096 | -31.91 | 2.01 | — | WARN |
| H4(c) | f2 | 4096 | -31.91 | 2.43 | — | FAIL |
| H2 | f4 | 2048 | 21.28 | 5.68 | 0.27 | FAIL |
| H4(c) | f4 | 2048 | 21.28 | 5.70 | — | FAIL |
| H2 | f4 | 4096 | 66.44 | 11.64 | 0.18 | FAIL |
| H4(c) | f4 | 4096 | 66.44 | 11.45 | — | FAIL |

---

## Summary

- PASS: 51
- WARN: 4 (warnings never count as passes)
- FAIL: 16
- **Overall: FAIL**

### Failing checks
- [a] f1/2048: residual=-33.84us
- [a] f1/4096: residual=-101.93us
- [a] f2/2048: residual=-14.68us
- [a] f2/4096: residual=-34.33us
- [a] f4/2048: residual=15.58us
- [a] f4/4096: residual=54.99us
- [b] f4/delta: measured delta 6.52 MB vs analytic 1.03 MB (accept [0.52, 1.55])
- [c] f4/4096: delta=-0.19us
- [H] H4c/f1/2048: |gap-(Δ+B)|=33.84us vs 2%·t_graph=4.39us
- [H] H4c/f1/4096: |gap-(Δ+B)|=101.93us vs 2%·t_graph=8.70us
- [H] H4c/f2/2048: |gap-(Δ+B)|=14.68us vs 2%·t_graph=8.51us
- [H] H4c/f2/4096: |gap-(Δ+B)|=34.33us vs 2%·t_graph=16.89us
- [H] H2/f4/2048: fraction=0.27
- [H] H4c/f4/2048: |gap-(Δ+B)|=15.58us vs 2%·t_graph=3.75us
- [H] H2/f4/4096: fraction=0.18
- [H] H4c/f4/4096: |gap-(Δ+B)|=54.99us vs 2%·t_graph=7.68us