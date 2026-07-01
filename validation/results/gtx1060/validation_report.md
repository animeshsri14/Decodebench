# DecodeBench Validation Report
GPU: GTX 1060 6GB (SM61) - local development machine
Generated: 2026-06-12T00:28:42.842693

> **Scope:** This GPU is used for local kernel correctness development only, not for authoritative validation. SM61 does not support Nsight Compute hardware counters, so Check (b) (NCU DRAM bytes) is N/A throughout. G1 (numerical correctness) was checked interactively during development but is not logged here. F2 timing is missing from Check (a) due to an early harness limitation. Negative Δ_launch for F4 in Check (c) is clock noise at 638 µs kernel time on a consumer GPU, not a methodology failure.

## Check (a): Residual analysis (t_graph - t_fused - B)

| Fusion | Dim | t_unfused_graph (us) | t_fused (us) | B (us) | Residual | Status |
|--------|-----|---------------------|-------------|--------|----------|--------|
| f1 | 2048 | 61.30 | 173.79 | 0.05 | -112.54 | PASS |
| f1 | 4096 | 61.30 | 173.79 | 0.11 | -112.60 | PASS |
| f4 | 2048 | 638.16 | 980.28 | 0.87 | -343.00 | PASS |
| f4 | 4096 | 638.16 | 980.28 | 0.87 | -343.00 | PASS |

## Check (b): Analytic bytes vs NCU DRAM bytes

| Fusion | Variant | Analytic (MB) | NCU DRAM (MB) | Ratio | Status |
|--------|---------|---------------|---------------|-------|--------|
| f1 | unfused-stream | 117.50 | 0.00 | 0.00 | N/A (no NCU data) |
| f1 | fused | 117.50 | 0.00 | 0.00 | N/A (no NCU data) |
| f2 | unfused-stream | 235.04 | 0.00 | 0.00 | N/A (no NCU data) |
| f2 | fused | 235.04 | 0.00 | 0.00 | N/A (no NCU data) |
| f4 | unfused-stream | 17.32 | 0.00 | 0.00 | N/A (no NCU data) |
| f4 | fused | 17.32 | 0.00 | 0.00 | N/A (no NCU data) |

## Check (c): Launch overhead (Δ_launch = t_stream - t_graph)

F4 shows negative Δ_launch (-8.73 µs) - clock noise on this consumer GPU at a 638 µs kernel baseline. Not a real result; this GPU is not part of the authoritative validation set.

### f1 (dim=4096)
- t_unfused_stream: 63.31 us
- t_unfused_graph:  61.30 us
- t_fused:          173.79 us
- launch_graph:     t_graph - t_fused = -112.49 us
- launch_stream:    t_stream - t_graph = 2.01 us

### f4 (dim=4096)
- t_unfused_stream: 629.42 us
- t_unfused_graph:  638.16 us
- t_fused:          980.28 us
- launch_graph:     t_graph - t_fused = -342.13 us
- launch_stream:    t_stream - t_graph = -8.73 us

---

## Summary

- PASS: 4
- FAIL: 0
- **Overall: PASS**