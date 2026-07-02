# DecodeBench - Is your LLM decode fusion really saving bytes, or just launch overhead?

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12+](https://img.shields.io/badge/CUDA-12%2B-76b900.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Quantify how much of your LLM decode-chain fusion gain comes from launch overhead vs. eliminated intermediate bytes.**

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
Verdict: LAUNCH-BOUND -> enable CUDA Graphs; hand-fusion is not worth the maintenance cost.
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
- **The launch-bound / byte-bound verdict classifies the analytic byte fraction**, not the empirically dominant source of gain. The `Verdict.dominant` field and render output additionally compare the *measured* Δ_launch against B, and the verdict cautions when the two disagree.
- **Byte accounting uses logical `tensor.nbytes` at stage boundaries.** Non-contiguous arguments/outputs, aliased or in-place outputs, and directly closure-captured tensor parameters are rejected with actionable errors. Dtype-conversion temporaries, tensors hidden inside containers/objects, and library-internal workspaces remain unmodeled; the internal-allocation warning is heuristic because allocator statistics are not DRAM traffic. Keep stages to single ops for the model to be meaningful.
- **Every non-final stage output must be consumed by a later stage** (by naming a parameter after the producing stage, or positionally). Unconsumed intermediates now raise an error instead of silently corrupting the eliminable-byte count.

---

## Pre-registered Hypotheses

These hypotheses were registered before any measurement runs:

- **H1:** F1 (RMSNorm→GEMV) and F2 (Gate/Up+SwiGLU) are **launch-bound** on all four GPUs: T4, L4, A100, RTX Pro 6000. Their measured Δ_launch explains ≥ 80% of the unfused-to-fused gap.
- **H2:** F4 (Attention scores+softmax+V) is **byte-bound** on all four GPUs. Its B estimate accounts for ≥ 80% of the unfused-to-fused gap.
- **H3:** Δ_launch (in microseconds) grows relative to t_graph as GPU bandwidth rises, since byte-elimination savings shrink with faster memory while launch overhead remains roughly constant across generations.
- **H4:** Three independent correctness checks pass within stated tolerances:
  - **(a)** Numerical correctness: every output satisfies an allclose-style mixed tolerance, `abs_error < 5e-2 OR rel_error < 2e-2` (FP16 storage, FP32 accumulation). F4 is checked independently against a scalar CPU attention reference as well as two GPU witnesses.
  - **(b)** Monotonicity: t_fused ≤ t_graph for all measured configurations. *Note: as of 2026-07-02 this holds on T4 for all three fusions, including F4, using the throughput-tuned split-KV FlashDecode kernel; the earlier single-block f4.cu (kept as a correctness reference) did not satisfy it. L4 results predate the tuned kernel.*
  - **(c)** Decomposition consistency: |(t_stream − t_fused) − (Δ_launch + B)| ≤ 2% of t_graph. *(Wording fixed 2026-07: the earlier text mixed baselines — Δ_launch + B decomposes the stream-to-fused gap, so the consistency check must be stated against t_stream.)*

---

## Validation status (as of 2026-07-02, revised)

**No GPU currently holds a valid PASS under the strengthened validation gates.** The 2026-07 revision (external review) made the pipeline fail-closed, restored the pre-registered hypotheses as enforced gates, made the F4 byte-delta gate two-sided, fixed the F2 byte model (it previously omitted the `u` intermediate), and equalized weight/KV cache residency across benchmark variants. Earlier reports claiming "33/33 PASS" were produced by weaker gates (favorable-direction violations reclassified as warnings, warnings counted as passes, missing data skipped) and by a benchmark with variant-dependent cache residency — **those results are retracted as validation evidence and all GPUs require re-runs**. The retained per-GPU data remains useful as raw measurements.

| GPU | Status | Notes |
|-----|--------|-------|
| L4 (SM89, Ada) | re-run required | Historical run predates the tuned split-KV F4 kernel, the F4 `--dim`=KV-length semantics, the F2 byte-model fix, and residency parity. |
| T4 (SM75, Turing) | re-run required; **H2 refuted on historical data** | On the 2026-07-02 data, B explains ~20% of the F4 fusion gap (pre-registered: ≥80%); the measured eliminated delta (6.8 MB) is >3× the analytic value (2.1 MB). Under the current gates these are FAILs, honestly reported. |
| A100 (SM80, Ampere) | pending | - |
| RTX Pro 6000 (SM120, Blackwell) | pending | - |

**Outcome notes and caveats (T4, 2026-07-02 run):**

- **H1 (F1/F2 launch-bound):** historical L4/T4 data has the expected directional signal, but it is not validation evidence after the accounting and residency fixes; fresh runs are required.
- **H4(b) monotonicity now holds for F4, non-vacuously:** the split-KV FlashDecode fused kernel beats the CUDA-graph baseline wall-clock at both KV lengths (L=2048: 180 vs 213 µs; L=4096: 370 vs 438 µs, ~16% faster). Both the fused kernel *and* the unfused attention baseline (`attn_scores`, `attn_v`) were tuned to the same coalesced-streaming idioms, so the comparison isolates fusion effects rather than kernel tuning quality.
- **H2 (F4 byte-bound) is refuted by the historical T4 decomposition.** The fused win exceeds Δ_launch + B (gap ~80 µs vs modeled ~22 µs at L=4096), and B accounts for ~17–20% rather than the pre-registered ≥80%. The likely contributors include elimination of low-parallelism stage boundaries and the split-KV algorithm's different scheduling. Under the current gates this is a FAIL of H2, H4(c), and Check (a), not a favorable warning.
- **Check (b) F4: the measured eliminated delta (97.6 − 90.8 = 6.8 MB) is about 6.6× the current analytic delta (~1.03 MB after subtracting split-KV partial-buffer traffic), which FAILS the two-sided gate.** A one-sided "delta ≥ analytic" gate validates nothing about model accuracy, so it was removed. Absolute totals remain diagnostics because the fused split-KV algorithm changes scheduling and cache behavior as well as intermediate traffic.
- **F4's declared-byte fraction is small by construction:** for single-query decode the eliminable intermediate (`O(L)`) versus KV traffic (`O(L·D)`) fixes the analytic fraction near **4/D ≈ 3%**, independent of sequence length. This is a byte ratio, not a guaranteed runtime fraction. The historical ~16% wall-clock win therefore cannot be attributed to bytes alone.

---

## Installation

```bash
pip install decodebench
```

For development (includes pytest and matplotlib):

```bash
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy. PyTorch is a soft dependency - required only for GPU code paths (lazy-imported so CPU-only analysis does not need a GPU).

---

## Usage

### CLI

```bash
decodebench demo f1          # Run built-in F1 demo (RMSNorm → GEMV)
decodebench demo f2          # Run built-in F2 demo (Gate/Up + SwiGLU)
decodebench demo f4          # Run built-in F4 demo (Attention scores+softmax+V)

decodebench profile --fusion F1 --dim 4096   # Profile a specific fusion and dimension
decodebench sweep --dims 2048 4096 8192       # Sweep dimensions, print analytic predictions
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
```

### Key Functions

| Function | Description |
|----------|-------------|
| `eliminable_bytes(traces)` | Sum of intermediate byte traffic that fusion can eliminate |
| `total_bytes(traces)` | Total unfused memory traffic |
| `compute_verdict(...)` | Classify fusion as launch-bound or byte-bound |
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
Verdict: LAUNCH-BOUND -> enable CUDA Graphs first; the declared byte fraction is below the configured fusion threshold.
```

For byte-bound fusions:
```
Verdict: BYTE-BOUND -> the declared byte fraction clears the threshold; benchmark a representative fused kernel before committing.
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
  note   = {Open-source tool. \url{https://github.com/animesh/decodebench}},
}
```

---

## License

MIT - see [LICENSE](LICENSE) for full text.
