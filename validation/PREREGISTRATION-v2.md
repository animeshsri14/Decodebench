# DecodeBench Pre-registration v2

**Registered:** 2026-07-02, before any A100, RTX Pro 6000, or post-overhaul L4
measurement. **Calibration dataset:** the T4 (SM75) run of 2026-07-02
(`results/t4/`), collected under the fail-closed methodology (residency
parity, unified F2 accounting, ≥30 samples/cell). T4 data was used to *design*
these gates and set their tolerances; it is therefore reported as the
calibration result. A100, RTX Pro 6000, and the L4 re-run are **confirmatory**:
they run these gates unchanged.

## Why v2 exists (v1 outcome summary)

v1 pre-registered that the graph-vs-fused gap is explained by byte
elimination alone (H2: "B accounts for ≥80% of the gap"; H4(c):
decomposition within 2% of t_graph). On T4 this was **refuted**: with both
the fused split-KV FlashDecode kernel and the unfused attention baseline
tuned to the same coalesced idioms, fused F4 beats the CUDA-graph baseline
by 21.3/66.6 µs (dim 2048/4096) while B is only 5.7/11.6 µs — B explains
~20% of the gap, not ≥80%. Conversely, fused F1/F2 are *slower* than the
graph baseline once cache residency is equalized. Both outcomes are real
findings the v1 model could not express: it had no term for
execution-structure effects.

## v2 decomposition

For each fusion chain and problem size, with medians over ≥30 trials:

```
t_graph − t_fused = B + S
```

- **Δ_launch = t_stream − t_graph** — CPU launch overhead recovered by CUDA
  Graphs. Unchanged from v1.
- **B = eliminable_bytes / achieved_bw** — proportional byte-time estimate
  (NOT an upper bound), achieved_bw = analytic total bytes / t_graph.
  Unchanged from v1.
- **S = (t_graph − t_fused) − B** — the **structural term**: all
  execution-structure effects of fusion beyond byte elimination.
  - **S > 0**: fusion removes structural cost — elimination of
    low-parallelism interleaved stages (e.g. the H-block softmax between two
    wide kernels) and of per-kernel-boundary drain/ramp that graph replay
    cannot remove.
  - **S < 0**: fusion adds structural cost — recomputation (e.g. fused F1
    re-derives the RMSNorm reduction), register pressure/occupancy loss.
- **τ_v** — per-round sum of isolated per-kernel GPU durations
  (NCU `gpu__time_duration.sum`, median across rounds). An independent
  instrument used **directionally only**: NCU replay flushes caches between
  kernels, inflating multi-kernel chains that enjoy inter-kernel L2 reuse in
  steady state (measured on T4: unfused F4 chain +26% vs wall-clock, fused
  +9%). The sign of τ_u − τ_f is robust to this bias; the magnitude is not.

## v2 gates (fail-closed; WARN never counts as PASS; nonzero exit on FAIL)

| Gate | Definition | Tolerance (T4-calibrated) |
|------|------------|---------------------------|
| G0 | Every timing cell (fusion × dim × variant) has ≥30 usable samples; every NCU cell (fusion × {unfused-stream, fused} × dim) has bytes AND kernel durations | — |
| G1 | correctness_ok = 1 on every row of every cell | abs 5e-2 AND rel 2e-2 |
| G2 | unfused GEMV ≥ 90% of cuBLAS bandwidth | — |
| (a) | sign(τ_u − τ_f) agrees with sign(t_graph − t_fused) | 5 µs near-zero indeterminacy band on either quantity |
| (b) F1/F2 | analytic vs NCU DRAM totals per dim | ratio within ±20% |
| (b) F4 | eliminated delta (unfused − fused) vs analytic delta (eliminable − split-KV partial traffic), **only when** analytic delta ≥ 5% of the smaller variant total | two-sided ±50%; below the 5% floor: below-resolution, no claim |
| (c) | Δ_launch ≥ −noise | noise = max(0.5%·t_graph, 2 µs) |
| H1-v2 | F1/F2: t_fused ≥ t_graph − noise AND S ≤ s_tol | noise as (c); s_tol = max(1%·t_graph, 3 µs) |
| H2-v2 | F4: t_fused < t_graph AND S > 0 AND S > B | — |

## Confirmatory predictions (A100 SM80, RTX Pro 6000 SM120, L4 SM89 re-run)

1. **H1-v2 holds on every GPU**: fused F1/F2 never beat the graph baseline
   beyond noise, and S ≤ 0 within tolerance (fusion of launch-bound chains
   adds recompute/occupancy cost; CUDA Graphs are the right tool there).
2. **H2-v2 holds on every GPU**: fused F4 beats the graph baseline, S > 0,
   and S > B — the F4 fusion win is structure-dominated, not byte-dominated.
3. **S grows with SM count relative to the gap's byte share.** The dominant
   structural contributor is the H=32-block softmax stage, which utilizes at
   most 32 SMs. Its relative cost — hence S — should be larger on A100
   (108 SMs) and RTX Pro 6000 (192 SMs) than on T4 (40 SMs). Directional
   prediction: S/B on A100 > S/B on T4 (T4 calibration values: 2.7 at
   dim 2048, 4.7 at dim 4096).
4. **The F4 NCU byte-delta remains below counter resolution** wherever the
   analytic delta (~0.5–1 MB) is under 5% of ~45–97 MB totals — i.e., on all
   planned GPUs at these problem sizes. Byte-elimination evidence for F4
   rests on B inside the decomposition, not on DRAM-counter deltas.

## What would falsify v2

- Fused F1/F2 beating the graph baseline beyond noise on any GPU (H1-v2).
- Fused F4 losing to the graph baseline, or S ≤ B, on any GPU (H2-v2).
- τ and wall-clock disagreeing in sign outside the 5 µs band (check a):
  would indicate the wall-clock verdict is a launch/boundary artifact.
- S/B on A100 ≤ S/B on T4 (prediction 3) — reported as a refuted prediction
  (does not gate the run's validity, since it compares across runs).

## Confirmatory outcomes

### L4 (SM89, Ada, 58 SMs) — 2026-07-02, `results/l4/validation_report.md`

Overall FAIL (74 PASS / 0 WARN / 2 FAIL); both FAILs are H2-v2. Gates ran
unchanged; verified by independent re-run of `compare.py` on the committed
data (identical tables, exit 1).

- **Prediction 1 (H1-v2 holds): CONFIRMED.** Fused F1/F2 slower than the
  graph baseline at both dims (S = −3.4 to −29.3 µs), τ sign corroborated.
- **Prediction 2 (H2-v2 holds): REFUTED.** F4 fused ties the graph baseline
  (141.90 vs 139.64 µs at dim 2048, within noise; 279.39 vs 279.40 µs at
  dim 4096) and S inverts: gap 2.26 = B 4.30 + S −2.04 (2048);
  gap −0.00 = B 8.46 + S −8.47 (4096). S ≤ 0 and S < B at both dims —
  this meets the pre-registered falsification criterion for H2-v2.
- **Prediction 3 (S/B grows with SM count; primary test A100 vs T4):
  pending A100, but the L4 data point contradicts the mechanism.**
  L4 S/B = −0.47 (2048) and −1.00 (4096) vs T4 calibration +2.7 / +4.7,
  despite 58 > 40 SMs.
- **Prediction 4 (F4 byte-delta below counter resolution): CONFIRMED.**
  Analytic deltas 0.52/1.03 MB against ~38–79 MB totals; recorded as
  below-resolution, no byte-delta claim, at both dims.

Interpretation: the T4 structure-bound F4 win is architecture-specific.
On Ada the fused split-KV F4 kernel's structural benefit is fully offset
(S < 0), leaving no wall-clock advantage. Recorded as a v2 negative
result; gates and tolerances unchanged per change control below.

### RTX Pro 6000 (SM120, Blackwell, 94 SMs visible — MIG 2g.48gb slice of a DC-2-48Q vGPU) — 2026-07-03, `results/rtx6000pro/validation_report.md`

Overall FAIL (42 PASS / 0 WARN / 12 FAIL). **Partial run: timing grid complete
(540 trials, all cells ≥ 30 samples), NCU collection impossible** — the vGPU
host profile blocks GPU performance counters (`ERR_NVGPUCTRPERM` even as
root); all 12 FAILs are G0 missing-NCU completeness cells. Gates ran
unchanged; checks (a), (b), and (H) could not execute for lack of the τ/byte
instrument. Wall-clock conclusions below are derived from the committed
`timing.csv` using the frozen v2 formulas (B is analytic from `t_graph` and
the byte model; no NCU input).

- **Prediction 1 (H1-v2 holds): REFUTED on timing (formal gate not run).**
  Fused F1 *beats* the graph baseline by 7.78 µs at dim 4096 (noise floor
  2 µs); fused F2 by 2.70/3.09 µs at 2048/4096. The "no fused win" condition
  fails at three of four F1/F2 cells; S = +2.55 to +7.75 µs exceeds the
  3 µs tolerance at f1/4096.
- **Prediction 2 (H2-v2 holds): REFUTED.** Fused F4 loses outright to the
  graph baseline — 74.72 vs 58.59 µs (2048), 114.91 vs 106.36 µs (4096).
  This meets the pre-registered wall-clock falsification criterion with no
  instrument caveat. Decomposition: gap −16.13 = B 1.77 + S −17.91 (2048);
  gap −8.55 = B 3.22 + S −11.77 (4096). S < 0 and S < B at both dims. The
  L4 refutation replicates on Blackwell: the inversion is not Ada-specific.
- **Prediction 3 (S/B grows with SM count): data point at 94 SMs, trend
  inverted.** S/B = −10.09 (2048) and −3.65 (4096) vs T4 +2.7/+4.7 (40 SMs)
  and L4 −0.47/−1.00 (58 SMs) — monotonically decreasing with SM count at
  both dims, the opposite slope from the registered mechanism. Note the
  planned full-die 192-SM test did not materialize: the VM exposes a 94-SM
  MIG slice.
- **Prediction 4 (F4 byte-delta below counter resolution): NOT MEASURABLE**
  (no NCU on this machine as provisioned).

Deviations: (1) first attempt (commit `ff90f8b`) crashed at CUDA init —
the guest driver had been installed with `--no-unified-memory`, omitting
`nvidia-uvm`; fixed at the driver level (uvm compat patch for the 7.0
kernel, module built and loaded), zero benchmark code changes. (2) NCU
blocked by the hypervisor-side vGPU profile — not fixable from the guest;
a passthrough or profiling-enabled instance is required for a full-gate
Blackwell run. Gates and tolerances unchanged per change control below.

## Provenance & change control

- v1 hypotheses and their T4 refutation are preserved in the README
  ("Validation status") and in `results/t4/validation_report.md`.
- Any further gate change after confirmatory data exists must be recorded
  the same way: a dated section here, with the pre-change outcome reported.
