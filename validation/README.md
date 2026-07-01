# DecodeBench Validation Suite

Reproducible CUDA validation kernels and harness for DecodeBench
("Quantifying Launch Overhead vs. Byte-Elimination in LLM Decode Fusion").

## Claim-Mapping Table

| Check    | Claim | Validation Method |
|----------|------------|-------------------|
| G1       | Kernel correctness | unfused vs CPU reference, max_abs < 5e-2, max_rel < 2e-2 |
| G2       | GEMV kernel >= 90% cuBLAS bandwidth | `calibrate --gate-g2` compares unfused FP16 GEMV to cublasGemmEx |
| (a)      | t_graph ≈ t_fused + B (residual ≤ 0) | `compare.py` computes residual_us = t_graph - t_fused - B |
| (b)      | Analytic byte model matches NCU DRAM counts within 20% | `compare.py` joins timing CSV + ncu metrics CSV |
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
│   │   │   ├── attn_scores.cu # Q·K^T attention scores
│   │   │   ├── softmax.cu     # Row softmax over L
│   │   │   └── attn_v.cu      # Attention·V projection
│   │   └── fused/
│   │       ├── f1.cu          # RMSNorm→GEMV (L2-resident norm)
│   │       ├── f2.cu          # GEMV→SwiGLU (dual dot product)
│   │       └── f4.cu          # FlashDecode (online softmax, no global scores)
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

# NCU collection only
bash validation/scripts/ncu_collect.sh

# Analysis only
python3 validation/analysis/compare.py \
  --timing-csv results/timing.csv \
  --ncu-csv results/ncu_metrics.csv \
  --output results/validation_report.md
```

### 5. Interpreting Results

- **G2 PASS**: GEMV kernel achieves ≥90% of cuBLAS bandwidth → kernels are efficient
- **Check (a) PASS**: t_graph - t_fused - B ≤ 0 → byte-elimination fully explains the fused speedup
- **Check (b) PASS**: Analytic byte model matches NCU DRAM counts within 10% → byte model is accurate
- **Check (c) PASS**: Launch overhead estimates are consistent → timing methodology is sound
- **Validation PASS**: All checks pass → all claims are validated

## Dimensions

| Parameter | Value | Description |
|-----------|-------|-------------|
| d (hidden) | 2048, 4096 | Model hidden dimension |
| d_in | = d | Input dimension for GEMV |
| d_out | 14336 | FFN intermediate dimension |
| H | 32 | Number of attention heads |
| L | 1024 | Sequence length (KV cache) |
| D | 128 | Head dimension |
| B | 1–8 | Batch size |
