# DecodeBench - Is your LLM decode fusion really saving bytes, or just launch overhead?

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12+](https://img.shields.io/badge/CUDA-12%2B-76b900.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Quantify how much of your LLM decode-chain fusion gain comes from launch overhead vs. eliminated intermediate bytes.**

When you fuse a pipeline of decode kernels (e.g., RMSNorm → GEMV), you observe a speedup. DecodeBench decomposes that speedup into two terms:

- **Δ_launch:** time saved by eliminating CUDA kernel launches - already captured by CUDA Graphs on any production engine.
- **B:** an analytic upper bound on the time that *could* be saved by eliminating intermediate DRAM traffic - and nothing more.

The distinction matters: CUDA Graphs can already eliminate launch overhead without any kernel fusion at all. Fusion that merely hides launches may be redundant. Fusion that eliminates intermediate bytes provides a genuine, additive gain beyond what CUDA Graphs alone can achieve.

---

## Example Output

Running `decodebench demo f1` on an L4 GPU produces:
```
CUDA Graphs eliminate 8.34 us here (95% CI [8.12, 8.56]) (measured).
Fusion can save at most 0.02 us from eliminable intermediate bytes (analytic byte ceiling).
Floor with graphs on (t_graph): 12.45 us.
Verdict: LAUNCH-BOUND - enable CUDA Graphs; hand-fusion is not worth the maintenance cost.
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

# A stage returns its output tensor. For every stage after the first, the
# first parameter is bound to the previous stage's output; the remaining
# parameters are looked up by name in the `inputs` dict passed to profile().
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
decomposes the stream-vs-graph gap into Δ_launch and the analytic byte ceiling B,
then classifies the fusion; `report.render()` is shorthand for printing it.

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

- **Linear chains only.** Each stage consumes exactly one predecessor's output. Branchy dataflow - QKV split, RoPE on a subset of heads - is not yet supported.
- **Single output per stage.** Each stage writes exactly one output tensor; multi-output stages are not modeled.
- **Byte accounting at stage boundaries.** Stage-internal temporaries that are read and written within a single stage are not tracked. A warning is emitted when a stage's internal footprint suggests significant hidden traffic.
- **B bounds only the byte-elimination component, not total fusion gain.** B = Bytes_eliminable / BW_measured captures the upper bound on time saved by eliminating intermediate reads and writes. It does *not* include computation reuse, occupancy improvements, reduced register pressure, or any other fusion effects. The residual `residual_us = (t_graph − t_fused) − B` isolates the unexplained portion.

---

## Pre-registered Hypotheses

These hypotheses were registered before any measurement runs:

- **H1:** F1 (RMSNorm→GEMV) and F2 (Gate/Up+SwiGLU) are **launch-bound** on all four GPUs: T4, L4, A100, RTX Pro 6000. Their measured Δ_launch explains ≥ 80% of the unfused-to-fused gap.
- **H2:** F4 (Attention scores+softmax+V) is **byte-bound** on all four GPUs. Its B bound accounts for ≥ 80% of the unfused-to-fused gap.
- **H3:** Δ_launch (in microseconds) grows relative to t_graph as GPU bandwidth rises, since byte-elimination savings shrink with faster memory while launch overhead remains roughly constant across generations.
- **H4:** Three independent correctness checks pass within stated tolerances:
  - **(a)** Numerical correctness: fused and unfused outputs match within 1×10⁻³ relative error (L∞ norm).
  - **(b)** Monotonicity: t_fused ≤ t_graph for all measured configurations. *Note: this does not hold for F4 - the shipped f4.cu is a correctness reference, not a throughput-tuned kernel; see validation status below.*
  - **(c)** Decomposition consistency: |(t_graph − t_fused) − (Δ_launch + B)| ≤ 2% of t_graph.

---

## Validation status (as of 2026-07-01)

Two of the four GPUs are measured; **A100 and RTX Pro 6000 runs will follow**, along with a throughput-tuned F4 kernel (see the F4 caveat below).

| GPU | Status | Notes |
|-----|--------|-------|
| L4 (SM89, Ada) | COMPLETE - 33/33 PASS | F1/F2 launch-bound; F4 byte-bound signal |
| T4 (SM75, Turing) | 32/33 | Sole failure: f4/fused Check (b), analytic-vs-NCU ratio 0.72 (outside ±20%) - T4's ~4 MB L2 thrashes the single-block FlashDecode reference. Hardware effect, not a bug; see caveat. |
| A100 (SM80, Ampere) | pending | - |
| RTX Pro 6000 (SM120, Blackwell) | pending | - |

**Outcome notes and caveats:**

- **H1 (F1/F2 launch-bound):** holds on L4 and T4.
- **H2 (F4 byte-bound):** the *analytic* byte ceiling B and the NCU byte ratio support the byte-bound classification, but there is **no wall-clock fusion-beats-graph demonstration** yet. The shipped `f4.cu` is a single-block online-softmax **correctness reference**, not a throughput-tuned FlashAttention, so fused F4 runs **~1.3–1.9× *slower*** than the CUDA-graph baseline. A throughput-tuned F4 kernel is planned for the follow-up runs.
- **H4(b) (monotonicity `t_fused ≤ t_graph`): does *not* hold for F4** for the reason above; it holds for F1/F2. The implemented harness gate is Check (a) - the residual `(t_graph − t_fused) − B ≤ 0` - which F4 satisfies, but note this passes **vacuously** for F4 (it is satisfied *because* fused is slower, and is not a wall-clock win).
- **F4's byte-elimination ceiling is small by construction:** for single-query decode the eliminable intermediate (scores+probs, `O(L)`) versus KV traffic (`O(L·D)`) fixes the eliminable fraction at **4/D ≈ 3% of runtime, independent of sequence length**. Increasing KV/context length does **not** create a crossover - even a fully tuned kernel wins at most ~3% at long context, likely within measurement noise.

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
| `summarize(data)` | Compute mean, median, std, and 95% CI |
| `bootstrap_diff_ci(a, b)` | Bootstrap 95% CI for the difference of two timing samples |

---

## Output Format

The `Verdict.render()` method produces a multi-line text block with:

```
CUDA Graphs eliminate X.XX us here (95% CI [lo, hi]) (measured).
Fusion can save at most X.XX us from eliminable intermediate bytes (analytic byte ceiling).
Floor with graphs on (t_graph): X.XX us.
Verdict: LAUNCH-BOUND - enable CUDA Graphs; hand-fusion is not worth the maintenance cost.
```

For byte-bound fusions:
```
Verdict: BYTE-BOUND - fusion provides a genuine gain beyond CUDA Graphs; worth the implementation cost.
```

When t_graph − t_fused < 2×B but B/t_graph > θ, the verdict is **MIXED**:
```
Verdict: MIXED - both launch and byte terms contribute; profile the fused kernel to confirm.
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
