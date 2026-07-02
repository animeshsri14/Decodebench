# DecodeBench Validation Suite

Reproducible CUDA validation kernels and harness for DecodeBench
("Quantifying Launch Overhead vs. Byte-Elimination in LLM Decode Fusion").

## Claim-Mapping Table

| Check    | Claim | Validation Method |
|----------|------------|-------------------|
| G1       | Kernel correctness | Every output passes `abs_error < 5e-2 OR rel_error < 2e-2`; F1/F2/F4 use independent CPU references |
| G2       | GEMV kernel >= 90% cuBLAS bandwidth | `calibrate --gate-g2` compares unfused FP16 GEMV to cublasGemmEx |
| (a)      | t_graph ≈ t_fused + B within ±2% of t_graph | `compare.py` checks `abs(t_graph - t_fused - B)` |
| (b)      | F1/F2 totals within 20%; F4 eliminated delta within a two-sided tolerance of the modeled delta | `compare.py` joins timing CSV + NCU metrics |
| (c)      | Launch overhead captured by graphs (Δ_launch > 0) | `compare.py` checks t_stream - t_graph vs 0 |

## Directory Structure

```
validation/
├── include/decodebench_val/
│   ├── kernel_args.h         # Unified kernel argument struct
│   ├── tensor.h              # FP16 tensor helpers
│   └── timing.h              # CUDA event timing wrappers
├── src/
│   ├── kernels/
│   │   ├── device/
│   │   │   └── warp_reduce.h # __shfl_xor_sync warp reductions, uint4 loads
│   │   ├── unfused/
│   │   │   ├── gemv.cu       # Warp-strided vectorized GEMV
│   │   │   ├── rmsnorm.cu     # Block-wide RMS normalization
│   │   │   ├── swiglu.cu      # SiLU-gated linear unit
│   │   │   ├── attn_scores.cu # Q·K^T scores (warp per score row, coalesced)
│   │   │   ├── softmax.cu     # Row softmax over L
│   │   │   └── attn_v.cu      # Attention·V (block per head×dim-group, coalesced)
│   │   └── fused/
│   │       ├── f1.cu          # RMSNorm→GEMV (L2-resident norm)
│   │       ├── f2.cu          # GEMV→SwiGLU (dual dot product)
│   │       └── f4.cu          # FlashDecode: split-KV partial+reduce (benchmarked)
│   │                          #   plus single-block correctness reference
│   ├── bench/
│   │   ├── calibrate.cu       # Null launch + G2 gate calibration
│   │   └── bench_variant.cu   # Full timing harness with graph capture
│   └── reference/             # CPU reference implementations
├── scripts/
│   ├── check_env.sh           # GPU environment validation
│   ├── ncu_collect.sh         # NCU metric collection
│   └── validate.sh            # Full pipeline orchestration
├── analysis/
│   └── compare.py             # Timing + NCU analysis → validation report
└── README.md                  # This file
```

## Kernel Conventions

- **CUDA C++17**, no external kernel libraries in measured paths
- **FP16 storage, FP32 accumulation**
- **128-bit vectorized loads** (uint4 reinterpret, 8 halves per load) on weight streams
- **Warp reduction** via `__shfl_xor_sync(0xffffffff, v, mask)` for mask ∈ {16,8,4,2,1}
- **No shared-memory atomics**
- **Block = 256 threads** (8 warps)
- **GEMV**: one warp per output row, grid = ceil(d_out / 8)
- **Target architectures**: SM 75 (T4, authoritative validation) and SM 61 (GTX 1060, local dev)

## Reproduction Steps

### 1. Prerequisites

```bash
# CUDA Toolkit >= 12.0
# Python >= 3.10 with numpy (for analysis)
# NVIDIA Nsight Compute (ncu) in PATH

# Verify environment
bash validation/scripts/check_env.sh
```

### 2. Build

```bash
mkdir -p validation/build && cd validation/build
cmake .. \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

For GTX 1060 local development:
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
```

### 3. Run Validation Pipeline

```bash
# Full pipeline: env check → G2 → timing grid → NCU → analysis
bash validation/scripts/validate.sh
```

### 4. Individual Steps

```bash
# G2 gate only
./validation/build/calibrate --gate-g2

# Single benchmark
./validation/build/bench_variant \
  --fusion f1 --variant fused --dim 4096 --batch 1 \
  --trials 30 --target-ms 20 --seed 42 --csv results/timing.csv

# NCU collection only (writes ncu_raw_*.csv, then aggregates them into
# ncu_metrics.csv via analysis/parse_ncu.py)
bash validation/scripts/ncu_collect.sh

# Analysis only
python3 validation/analysis/compare.py \
  --timing-csv results/timing.csv \
  --ncu-csv results/ncu_metrics.csv \
  --output results/validation_report.md
```

### 5. Interpreting Results

- **G2 PASS**: GEMV kernel achieves ≥90% of cuBLAS bandwidth → kernels are efficient
- **Check (a) PASS**: `abs(t_graph - t_fused - B) ≤ 2% * t_graph`. Large residuals in either direction fail because the model does not explain the measured graph-to-fused difference.
- **Check (G0) PASS**: every expected (fusion, dim, variant) timing cell and NCU cell is present with usable data. Missing data is a FAIL — the pipeline is fail-closed.
- **Check (b) PASS**: F1/F2 — analytic byte model matches NCU DRAM totals within 20%. F4 — measured eliminated delta lies within a two-sided tolerance of the analytic delta (eliminable bytes minus modeled split-KV partial traffic). A delta far above the model FAILs just like one below it.
- **Check (c) PASS**: Launch overhead estimates are consistent → timing methodology is sound
- **Check (H) PASS**: the pre-registered hypotheses (H1/H2 ≥80% attribution, H4(c) 2% decomposition consistency) hold as gates; a refuted hypothesis is a FAIL, not a warning.
- **Validation PASS**: every registered check passes. Any FAIL yields `FAIL`; any WARN without a FAIL yields `INCOMPLETE` and a nonzero exit status.

## Dimensions

| Parameter | Value | Description |
|-----------|-------|-------------|
| d (hidden) | 2048, 4096 | F1/F2: model hidden dimension (`--dim`) |
| d_in | = d | Input dimension for GEMV |
| d_out | 14336 | FFN intermediate dimension |
| H | 32 | Number of attention heads (fixed) |
| L | 2048, 4096 | F4: KV-cache length (`--dim`; multiple of 128) |
| D | 128 | Head dimension (fixed) |
| B | 1 | Batch. Only 1 is accepted: no kernel is batched, and bench_variant rejects `--batch != 1` so unbatched runs can never be labeled as batch sweeps. |

`--dim` semantics per fusion: for F1/F2 it is the hidden dimension; for F4 it
is the KV-cache length L, so F4 sweeps vary the real problem size.
**History note:** results committed before 2026-07-02 ran F4 at a fixed
L=1024 regardless of the `dim` CSV column (it was a label only), and used
the historical single-block fused kernel and untuned unfused
attention kernels — do not compare their F4 rows against newer runs.

### F4 fused variant

The benchmarked "fused" F4 is a split-KV FlashDecode: `f4_partial_kernel`
(grid = H × n_splits, one KV tile of 128 rows per block, online softmax,
K/V read with the same coalesced idioms as the tuned unfused kernels)
followed by `f4_reduce_kernel` (log-sum-exp merge). It is **2 launches**,
vs 3 in the unfused chain; scores/probs never touch global memory.
`DECODEBENCH_F4_SPLITS=<n>` may be used for standalone tuning experiments, but
the official validation scripts unset it because the analytic delta models the
default split count. The historical single-block `f4_kernel` remains a second
GPU witness; an independent scalar CPU attention implementation is the G1
correctness authority.

The split-KV fused path is not a fusion-only controlled transformation: it
also changes parallel decomposition, reduction scheduling, occupancy, and
cache behavior. F4 timing and NCU deltas therefore validate this concrete
implementation, not a causal claim that every difference comes solely from
eliminating scores/probabilities. Unexplained residuals fail validation.
