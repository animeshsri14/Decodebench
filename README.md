# DecodeBench - Is your LLM decode fusion really saving bytes, or just launch overhead?

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12+](https://img.shields.io/badge/CUDA-12%2B-76b900.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A CUDA benchmarking and analysis framework for separating graph-recoverable launch overhead from estimated intermediate-byte cost, with cross-architecture experiments that falsified the initial fusion hypothesis.**

When you fuse a pipeline of decode kernels (e.g., RMSNorm → GEMV), you observe a speedup. DecodeBench decomposes that speedup into two terms:

- **Δ_launch:** time saved by eliminating CUDA kernel launches - already captured by CUDA Graphs on any production engine. Measured.
- **B:** a *proportional byte-time estimate* — the time the eliminable intermediate DRAM traffic would take at the workload's average byte throughput (total declared bytes / t_graph). **B is not a mathematical upper bound**: the linear-scaling assumption fails for launch-, latency-, compute-, or occupancy-bound kernels, cache-resident traffic, and serialized stage boundaries, and measured fusion gains can exceed B.

The distinction matters: CUDA Graphs can already eliminate launch overhead without any kernel fusion at all. Fusion that merely hides launches may be redundant. Fusion that eliminates intermediate bytes provides a genuine, additive gain beyond what CUDA Graphs alone can achieve.

---

## Example Output

Running `decodebench demo f1` on an L4 GPU produces:
```
CUDA Graphs eliminate 8.34 us here (95% CI [8.12, 8.56]) (measured).
Eliminable intermediate bytes correspond to ~0.02 us at this workload's average byte throughput (analytic proportional estimate, NOT a strict bound).
Floor with graphs on (t_graph): 12.45 us.
Dominant term: launch overhead (measured) (delta_launch 8.34 us vs B 0.02 us).
Verdict: LOW-BYTE-OPPORTUNITY -> enable CUDA Graphs first; the declared byte fraction is below the configured fusion threshold.
```

---

## Quickstart

```bash
pip install decodebench
decodebench demo f1
```

This runs the built-in F1 (RMSNorm → GEMV) demo on your GPU and prints a verdict block.

---

## Defining Your Own Chain

```python
import torch
import decodebench as db

seq = db.Sequence("rmsnorm-gate")

# A stage returns its output tensor. Parameters resolve by name, in order:
# (1) an earlier stage's output whose STAGE NAME matches the parameter name,
# (2) an entry in the `inputs` dict, then (3) for the first parameter of a
# non-first stage only, the previous stage's output (positional fallback).
# Name-based binding lets non-chain dataflow (e.g. F2's gate/up feeding
# SwiGLU) declare every real intermediate to the byte model.
@seq.stage
def rmsnorm(x, g):
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + 1e-6)).half() * g

@seq.stage
def gate_proj(xh, W):          # xh is the rmsnorm output; W comes from inputs
    return torch.nn.functional.linear(xh, W)

d = 4096
inputs = {
    "x": torch.randn(1, d, dtype=torch.float16, device="cuda"),
    "g": torch.randn(d, dtype=torch.float16, device="cuda"),
    "W": torch.randn(d, d, dtype=torch.float16, device="cuda"),
}

report = seq.profile(inputs, trials=30, warmup=50)
print(report.render())
```

`seq.profile()` times the chain twice - once as plain stream launches, once
replayed from a captured CUDA Graph - and returns a `Report`. `report.verdict()`
measures the stream-vs-graph launch term Δ_launch and estimates the proportional
time associated with eliminable bytes (B), then classifies the analytic byte
fraction; `report.render()` is shorthand for printing it.

---

## Built-in Demos

| Demo | Description |
|------|-------------|
| **F1** | RMSNorm → GEMV |
| **F2** | Gate projection + Up projection + SwiGLU (SiLU + element-wise multiply) |
| **F4** | Attention scores (Q·Kᵀ) + softmax + V multiplication |

Run any demo:

```bash
decodebench demo f1
decodebench demo f2
decodebench demo f4
```

---

## Limitations

- **Single-output stages in a fixed execution order.** Stages may consume any earlier stage's output by name, including fan-in and fan-out, and consumer multiplicity is included in eliminable-byte accounting. Each stage still writes exactly one tensor; multi-output stages — QKV split or RoPE returning rotation pairs — are not supported.
- **Byte accounting at stage boundaries.** Stage-internal temporaries that are read and written within a single stage are not tracked. A warning is emitted when a stage's internal footprint suggests significant hidden traffic.
- **B estimates only the byte-elimination component, not total fusion gain — and it is an estimate, not a bound.** B = Bytes_eliminable / BW_avg (with BW_avg = total declared bytes / t_graph) assumes runtime scales linearly with declared bytes and that eliminated intermediates see the same effective bandwidth as dominant weight/KV traffic. It does *not* include computation reuse, occupancy improvements, reduced register pressure, elimination of inter-kernel serialization, or any other fusion effects; measured gains can exceed B (observed on T4 F4). The residual `residual_us = (t_graph − t_fused) − B` isolates the unexplained portion.
- **The low-byte-opportunity / material-byte-opportunity verdict classifies the analytic byte fraction**, not the empirically dominant source of gain — and deliberately not the actual bottleneck: a chain below the threshold could still be compute-, latency-, or occupancy-bound. (Pre-2026-07-03 versions labeled these "launch-bound"/"byte-bound", which overstated the diagnosis.) The `Verdict.dominant` field and render output additionally compare the *measured* Δ_launch against B, and the verdict cautions when the two disagree.
- **Byte accounting uses logical `tensor.nbytes` at stage boundaries.** Non-contiguous arguments/outputs, aliased or in-place outputs, and directly closure-captured tensor parameters are rejected with actionable errors. Dtype-conversion temporaries, tensors hidden inside containers/objects, and library-internal workspaces remain unmodeled; the internal-allocation warning is heuristic because allocator statistics are not DRAM traffic. Keep stages to single ops for the model to be meaningful.
- **Every non-final stage output must be consumed by a later stage** (by naming a parameter after the producing stage, or positionally). Unconsumed intermediates now raise an error instead of silently corrupting the eliminable-byte count.

---

## Scope — what this project measures and what it does not

- **Timings are steady-state microchain latency**, not single-token or end-to-end decode latency: each trial averages hundreds of back-to-back invocations of the same chain at fixed shapes. Production effects — scheduler behavior, dynamic batching, CUDA Graph argument updates, CPU framework overhead — are outside the measurement.
- **The validation study is a focused case study, not a general decode-fusion benchmark.** It covers three fusion patterns at batch 1, FP16, H=32 heads, head dim 128, contiguous KV cache, L ∈ {2048, 4096}. GQA/MQA, paged KV, quantization, ragged/continuous batching, speculative decoding, MoE, and tensor parallelism are all out of scope.
- **The cross-architecture results characterize a fixed implementation, not tuned fusion economics.** The fused F4 kernel's split configuration was tuned on T4 and deliberately frozen for the confirmatory runs (pre-registration discipline). The Blackwell loss therefore shows that *this fixed, T4-tuned fused kernel* does not transfer — not that a per-architecture-tuned fused kernel could not win there. An equal-tuning-budget study is future work.
- **The Python package and the CUDA validation suite are separate surfaces.** The package (`decodebench` CLI + `Sequence` API) profiles *your* PyTorch stage chains; the validation suite (`validation/`) benchmarks the study's handwritten CUDA kernels with its own workload geometry (e.g., validation F4 treats `dim` as KV length with H=32/D=128 fixed, while the Python F4 demo treats `dim` as hidden size). GPU validation validates the paper's kernels, not the Python demos.

---

## Pre-registered Hypotheses

**Current registration: v2** (`validation/PREREGISTRATION-v2.md`, 2026-07-02, registered before any A100/RTX Pro 6000/post-overhaul-L4 measurement). v2 decomposes the fusion gap as **t_graph − t_fused = B + S**, where **S** is the **unexplained residual**: everything the proportional byte estimate B does not capture — execution-structure effects, occupancy and register-pressure differences, recompute, cache behavior, byte-model error, and measurement error. The registration named S the "structural term" and hypothesized a mechanism (elimination of low-parallelism interleaved stages and kernel-boundary drain when S > 0; recompute/occupancy cost when S < 0); that name is preserved in the frozen pre-registration text, but S is a residual, not a measured mechanism, and the mechanism story remains a hypothesis. The independent NCU per-kernel-duration instrument corroborates S's *sign* only (check a) — it confirms which implementation is faster, not why.

- **H1-v2:** F1 (RMSNorm→GEMV) and F2 (Gate/Up+SwiGLU) are **launch-bound / fusion-not-worthwhile** on all four GPUs: fused never beats the graph baseline beyond noise, and S ≤ 0 within tolerance — fusing launch-bound chains adds recompute/occupancy cost; CUDA Graphs are the right tool.
- **H2-v2:** F4 (Attention scores+softmax+V) fusion is **structure-bound** on all four GPUs: fused beats the graph baseline (t_fused < t_graph), S > 0, and **S > B** — the win is dominated by structural elimination, not byte elimination.
- **Cross-GPU prediction:** S/B for F4 grows with SM count (the dominant structural contributor, the H=32-block softmax, underutilizes larger GPUs more). T4 calibration: S/B = 2.7 (dim 2048), 4.7 (dim 4096); prediction: larger on A100 (108 SMs) and RTX Pro 6000 (192 SMs).
- **H4(a) numerical correctness** (unchanged): every output satisfies `abs_error < 5e-2 OR rel_error < 2e-2` (FP16 storage, FP32 accumulation); F4 checked against a scalar CPU reference plus two GPU witnesses.

**Retired v1 hypotheses** (registered before measurement; outcome recorded, gates retired 2026-07-02): v1-H2 claimed B alone explains ≥ 80% of the F4 gap — **refuted on T4** (B explains ~20%; the rest is S). v1-H4(c) claimed the gap decomposes into Δ_launch + B within 2% — refuted with it. v1-H1's "Δ_launch explains the gap" framing became undefined once residency parity made fused F1/F2 *slower* than the graph baseline (there is no positive gap to attribute); H1-v2 states that outcome directly. The refutation data is preserved in `results/t4/` and motivated the v2 registration.

---

## Validation status (as of 2026-07-03, pre-registration v2)

**T4 (SM75) is fully validated under the v2 fail-closed gates: 73 PASS / 3 INDETERMINATE / 0 WARN / 0 FAIL, Overall PASS** (indeterminate = check ran but could establish no claim either way — below counter resolution or an instrument inside the noise band; introduced 2026-07-03, see PREREGISTRATION-v2 change control; previously these 3 recorded vacuous PASSes) (`results/t4/validation_report.md`). The run uses the corrected methodology throughout — residency parity across variants, unified F2 gate/up/swiglu accounting, ≥30 samples per cell, per-dim NCU byte *and* per-kernel-duration collection — and the v2 decomposition `t_graph − t_fused = B + S` with directional two-instrument corroboration. T4 is the **calibration dataset** for v2 (its data set the gate tolerances); A100, RTX Pro 6000, and the L4 re-run are **confirmatory** and run the same gates unchanged.

| GPU | Status | Headline result |
|-----|--------|-----------------|
| T4 (SM75, Turing, 40 SMs) | **VALIDATED — 73 PASS / 3 INDETERMINATE / 0 FAIL (v2, calibration)** | F4 fused beats the CUDA-graph baseline: 166.3 vs 187.6 µs (L=2048) and 317.5 vs 384.2 µs (L=4096, **17% faster**). Decomposition: gap 66.6 µs = B 11.6 µs (bytes) + S 55.0 µs (structure) — **structure-bound, S/B = 4.7**. F1/F2 fused are slower than the graph baseline (S = −34 to −102 µs): fusion not worthwhile; CUDA Graphs suffice. G1 correctness 18/18; G2 GEMV 108.6% of cuBLAS. |
| L4 (SM89, Ada, 58 SMs) | **H2-v2 REFUTED — 69 PASS / 5 INDETERMINATE / 2 FAIL, Overall FAIL (v2, confirmatory)** | F4 fused does **not** beat the graph baseline: 139.6 vs 141.9 µs (L=2048, within noise) and 279.40 vs 279.39 µs (L=4096, exact tie). The structural term inverts: gap 2.26 µs = B 4.30 + S −2.04 (L=2048); gap −0.00 µs = B 8.46 + S −8.47 (L=4096) — **S/B = −0.47 and −1.00 vs T4's +2.7/+4.7**, refuting the S-grows-with-SM-count prediction (58 vs 40 SMs). H1-v2 holds (fused F1/F2 slower, S = −3.4 to −29.3 µs). G1 correctness 18/18; G2 GEMV 100.6% of cuBLAS; τ sign-corroboration 3 corroborated / 3 indeterminate (instrument inside noise band — 2026-07-03 accounting; was reported 6/6). The run is clean — the FAIL is the pre-registered hypothesis being falsified (`results/l4/validation_report.md`). |
| A100 (SM80, Ampere, 108 SMs) | confirmatory, pending | v2 prediction: H1-v2/H2-v2 hold; F4 S/B > 4.7 (larger SM count → larger structural share). |
| RTX Pro 6000 (SM120, Blackwell, 94 SMs visible — MIG 2g.48gb slice of a DC-2-48Q vGPU) | **H2-v2 REFUTED on wall-clock; PARTIAL — 42 PASS / 12 FAIL, Overall FAIL (v2, confirmatory, timing-only)** | F4 fused **loses outright** to the graph baseline: 74.7 vs 58.6 µs (L=2048, 27% slower) and 114.9 vs 106.4 µs (L=4096) — the pre-registered wall-clock falsification criterion for H2-v2, met without any NCU input. Decomposition from the frozen byte model: gap −16.13 µs = B 1.77 + S −17.91 (L=2048); gap −8.55 µs = B 3.22 + S −11.77 (L=4096) — **S/B = −10.09 and −3.65**: the S/B-vs-SM-count trend is now monotonically *decreasing* (T4 +2.7/+4.7 @ 40 SMs → L4 −0.47/−1.00 @ 58 → −10.09/−3.65 @ 94). F1/F2 invert the other way: fused F1 *beats* the graph baseline by 7.78 µs at L=4096 (beyond the 2 µs noise floor; F2 by 2.7–3.1 µs), so H1-v2's "no fused win" condition also fails here. G1 correctness 18/18; G2 GEMV 101.1% of cuBLAS. All 12 FAILs are G0 missing-NCU cells: the vGPU host profile blocks GPU performance counters, so τ corroboration and byte gates could not run (`results/rtx6000pro/validation_report.md`). |

**Outcome notes (T4, 2026-07-02, v2):**

- **H2-v2 (F4 structure-bound) holds:** fused wins wall-clock at both KV lengths, S > 0, and S > B (2.7× at L=2048, 4.7× at L=4096). The independent NCU per-kernel-duration instrument corroborates the direction (isolated fused kernels are 55–136 µs faster in aggregate) — direction only; it does not establish the mechanism. The *hypothesized* contributor is elimination of the H=32-block softmax stage and kernel-boundary drain — effects graph replay cannot remove — but S is a residual and this attribution is not causally identified (and the L4/Blackwell inversions below argue against it being the whole story).
- **H1-v2 (F1/F2 fusion-not-worthwhile) holds:** with cache residency equalized, fused F1/F2 are slower than both baselines (S = −33.8/−101.9 µs for F1, −14.7/−34.3 µs for F2). The earlier apparent fused F1/F2 wins were an L2-residency artifact of the pre-overhaul benchmark. Practical guidance stands: for launch-bound chains, use CUDA Graphs, don't hand-fuse.
- **v1-H2 ("byte-bound", B ≥ 80% of gap) is refuted and retired:** B explains ~20% of the F4 gap. F4's eliminable-byte fraction is structurally small (≈4/D ≈ 3% of traffic, independent of L) — the wall-clock win is real but comes from structure, not bytes. Do not cite the 17% as a byte-elimination result.
- **F4 NCU byte-delta is below counter resolution on T4** (analytic delta 0.5–1.0 MB vs ~46–97 MB totals with a uniform ~1.3–1.4× counter excess on KV streams, both variants): recorded as below-resolution, no byte-delta claim. Absolute totals stay in the report as diagnostics. F1/F2 byte totals gate normally and pass at ratios 0.84–0.89.
- **Known operational caveats (unchanged):** GPU clocks not locked (`--clock-control none`; timings were stable across run halves), variants not interleaved within a run.

**Outcome notes (L4, 2026-07-02, v2 confirmatory):**

- **H2-v2 is refuted on L4.** The F4 fused win observed on T4 does not transfer: on L4 the fused kernel merely ties the graph baseline, and S is negative at both KV lengths (fusion adds structural cost on this part instead of removing it). Under pre-registration v2's falsification criteria ("fused F4 losing to the graph baseline, or S ≤ B, on any GPU"), this is a confirmatory refutation — the T4 structure-bound characterization of F4 is architecture-specific, not general.
- **The S/B-grows-with-SM-count mechanism is contradicted by the L4 data point.** L4 has more SMs than T4 (58 vs 40), so the H=32-block-softmax argument predicted a larger structural share; instead S/B fell from +2.7/+4.7 to −0.47/−1.00. The A100 run (108 SMs, prediction 3's primary test) remains worth running, but the proposed mechanism cannot be the whole story.
- **Everything else replicated:** H1-v2 held (fused F1/F2 slower than the graph baseline, as on T4), all 18 correctness checks passed, τ/wall-clock sign corroboration held wherever both instruments were outside the noise band (3 corroborated, 3 indeterminate under the 2026-07-03 accounting), F1/F2 byte totals gated at ratios 0.87–0.90, and the F4 byte-delta was below counter resolution as predicted (prediction 4). The gates ran unchanged; the FAIL is the hypothesis, not the instrument. Environment: driver 595.71.05, CUDA 12.9.86, g++-14.3.0; NCU collection required a sudo rerun of `ncu_collect.sh` (filesystem permissions), no code changes.

**Outcome notes (RTX Pro 6000, 2026-07-03, v2 confirmatory — partial, timing-only):**

- **The L4 refutation of H2-v2 replicates on Blackwell, and harder.** On L4 the fused F4 kernel merely tied the graph baseline; here it loses outright at both KV lengths (74.7 vs 58.6 µs at L=2048; 114.9 vs 106.4 µs at L=4096). This meets pre-registration v2's wall-clock falsification criterion ("fused F4 losing to the graph baseline … on any GPU") directly from the timing grid, with no NCU input. The T4 structure-bound F4 win is now contradicted on two independent post-Turing architectures: the inversion is not Ada-specific.
- **Prediction 3's mechanism is not just refuted — the trend runs the other way.** Applying the frozen v2 decomposition (B is analytic: eliminable bytes ÷ achieved bandwidth from `t_graph`; no NCU needed), S/B at L=2048 falls monotonically with SM count: +2.7 (T4, 40 SMs) → −0.47 (L4, 58 SMs) → −10.09 (here, 94 visible SMs); at L=4096: +4.7 → −1.00 → −3.65. The H=32-block-softmax argument predicted the opposite slope. An A100 run (108 SMs) would now be testing a dead prediction; it remains useful only as an Ampere data point.
- **H1-v2's conditions also fail on this part — in the opposite, fused-wins direction.** Fused F1 beats the graph baseline by 7.78 µs at L=4096 (noise floor 2 µs) and fused F2 by 2.70/3.09 µs at L=2048/4096. These are timing-derived observations (the formal check (H) gate needs the NCU τ instrument, unavailable here — see below), but under the frozen thresholds the "no fused win" condition is violated. Fusion economics on Blackwell appear inverted relative to Turing in both directions: F4 fusion hurts, F1/F2 fusion mildly helps.
- **Why the run is partial, and what the 12 FAILs are.** The VM exposes the GPU as an NVIDIA vGPU (`DC-2-48Q`) with MIG enabled; the single 2g.48gb instance exposes 94 SMs, not the full die, and the vGPU host profile blocks GPU performance counters (`ERR_NVGPUCTRPERM` even as root — a hypervisor-side `enable_profiling` setting no guest configuration can override). NCU collection is therefore impossible on this machine as provisioned; all 12 FAILs in the report are G0 missing-NCU completeness cells, and check (a) τ corroboration, check (b) byte gates, and the formal check (H) rows could not run. G1 correctness is 18/18 and G2 GEMV is 101.1% of cuBLAS. Gates and tolerances unchanged per change control.
- **Machine-setup fault found and fixed (no benchmark code changes).** The first run attempt (commit `ff90f8b`) crashed at CUDA init with `unknown error`: the guest driver had been installed with `./nvidia-installer --no-unified-memory`, which skips the `nvidia-uvm` kernel module CUDA requires — and uvm then failed to build against the 7.0 GCP kernel (`zone_device_page_init` signature change). Fixed by patching the driver's uvm compat shim (guest driver source only), building and loading `nvidia-uvm.ko`, and persisting the fix in the DKMS tree. DecodeBench itself needed zero changes; the T4/L4-validated code ran unmodified. Environment: driver 580.126.09 (vGPU guest), CUDA 12.9.86, g++-14.3.0, arch sm_120.

---

## Installation

```bash
pip install decodebench
```

For development (includes pytest and matplotlib):

```bash
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy. PyTorch is a soft dependency - required only for GPU code paths (lazy-imported so CPU-only analysis does not need a GPU). To install it alongside the package:

```bash
pip install "decodebench[gpu]"
```

---

## Usage

### CLI

```bash
decodebench demo f1          # Run built-in F1 demo (RMSNorm → GEMV)
decodebench demo f2          # Run built-in F2 demo (Gate/Up + SwiGLU)
decodebench demo f4          # Run built-in F4 demo (Attention scores+softmax+V)

decodebench demo f1 --dim 2048 --batch 1 --trials 30   # Demo with explicit options
decodebench profile my_chain.py:build_fn                # Profile a user-defined Sequence
                                                        #   (build_fn returns (seq, inputs))
decodebench sweep f1 --batch 1,2,4,8 --dim 4096         # Sweep batch sizes for a demo
```

All subcommands support `--dry-run` to preview without executing GPU code.

### Python API

```python
import decodebench as db

# Pre-implementation estimation (no fused kernel needed)
from decodebench.bytes_model import StageTrace, eliminable_bytes, total_bytes

traces = [
    StageTrace("rms_norm", reads=[8192], write=8192, is_final=False),
    StageTrace("gate_proj", reads=[8192, 33554432], write=8192, is_final=True),
]
elim = eliminable_bytes(traces)
total = total_bytes(traces)
print(f"Eliminable: {elim:,d} / {total:,d} bytes ({elim/total*100:.2f}%)")

# Full profiling pipeline
seq = db.Sequence("my-fusion")
# ... register stages with @seq.stage ...
inputs = {"x": torch.randn(1, 4096, dtype=torch.float16, device="cuda")}  # example
report = seq.profile(inputs, trials=30, warmup=50)
print(report.render())          # report.verdict() returns the Verdict object

# Controlled fused comparison: pass your fused implementation and DecodeBench
# times it on the same input tensors, interleaved with the unfused stream and
# graph variants, after checking its output against the unfused chain
# (abs < 5e-2 or rel < 2e-2 elementwise — same tolerance as the CUDA
# validation harness). Report.fused_us and Verdict.t_fused are populated.
def my_fused(inputs):            # same inputs dict, returns the final tensor
    return my_fused_kernel(inputs["x"])

report = seq.profile(inputs, trials=30, warmup=50, fused=my_fused)
```

### Key Functions

| Function | Description |
|----------|-------------|
| `eliminable_bytes(traces)` | Sum of intermediate byte traffic that fusion can eliminate |
| `total_bytes(traces)` | Total unfused memory traffic |
| `compute_verdict(...)` | Classify the eliminable-byte opportunity (low vs material) |
| `report.verdict()` | Return the `Verdict` object from a profiling `Report` |
| `summarize(data)` | Compute median, p25, p75, and IQR |
| `bootstrap_diff_ci(a, b)` | Bootstrap 95% CI for the difference of two timing samples |

---

## Output Format

The `Verdict.render()` method produces a multi-line text block with:

```
CUDA Graphs eliminate X.XX us here (95% CI [lo, hi]) (measured).
Eliminable intermediate bytes correspond to ~X.XX us at this workload's average byte throughput (analytic proportional estimate, NOT a strict bound).
Floor with graphs on (t_graph): X.XX us.
Dominant term: launch overhead (measured) (delta_launch X.XX us vs B X.XX us).
Verdict: LOW-BYTE-OPPORTUNITY -> enable CUDA Graphs first; the declared byte fraction is below the configured fusion threshold.
```

For fusions whose declared byte fraction clears the threshold:
```
Verdict: MATERIAL-BYTE-OPPORTUNITY -> the declared byte fraction clears the threshold; benchmark a representative fused kernel before committing.
```

---

## Citing

Citation:

```bibtex
@misc{decodebench2026,
  title  = {DecodeBench: A Quantitative Framework for Decomposing Launch
            Overhead vs.\ Byte-Elimination in LLM Decode-Chain Fusion},
  author = {Animesh},
  year   = {2026},
  note   = {Open-source tool. \url{https://github.com/animeshsri14/Decodebench}},
}
```

---

## License

MIT - see [LICENSE](LICENSE) for full text.
