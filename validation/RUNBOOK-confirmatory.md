# Confirmatory-run runbook (L4 next, then A100 / RTX Pro 6000)

Purpose: reproduce the full validation pipeline on a confirmatory GPU under
pre-registration v2 (`PREREGISTRATION-v2.md`) with **zero code changes**. If a
gate fails, that is a result — report it; do not adjust gates (change control
is described in the pre-registration).

## 0. One-time machine setup

```bash
git clone https://github.com/animeshsri14/Decodebench && cd Decodebench
pip install -e ".[dev]"            # or --user --break-system-packages on PEP-668 systems
python3 -m pytest -m "not gpu" -q  # expect: all passed, 1 skipped (torch-gated)
```

Gotchas seen on prior boxes:
- **gcc > 14**: CUDA ≤ 12.9 rejects it and CMake silently reports CPU-only.
  Build with `NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-14"`. Verify the CMake
  output says "CUDA found".
- **`pip install -e .` is required** before pytest or `decodebench` imports
  fail at collection.

## 1. Build (choose arch: L4=89, A100=80, RTX Pro 6000=120)

```bash
mkdir -p validation/build-<gpu> && cd validation/build-<gpu>
NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-14" cmake .. \
  -DCMAKE_CUDA_ARCHITECTURES=<arch> -DCMAKE_BUILD_TYPE=Release
NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-14" cmake --build . -j$(nproc)
cd ../..
```

## 2. Full pipeline

```bash
BUILD_DIR=$PWD/validation/build-<gpu> bash validation/scripts/validate.sh
```

- The timing grid needs ~30–45 min. `validate.sh` and `compare.py` are
  fail-closed: nonzero exit on any FAIL or missing cell.
- **NCU needs root** on GCP images (`ERR_NVGPUCTRPERM`). If step 4 fails,
  re-run collection alone and then the analysis:

```bash
sudo env "PATH=/usr/local/cuda/bin:$PATH" BUILD_DIR=$PWD/validation/build-<gpu> \
  bash validation/scripts/ncu_collect.sh
python3 validation/analysis/compare.py \
  --timing-csv validation/results/timing.csv \
  --ncu-csv validation/results/ncu_metrics.csv \
  --output validation/results/validation_report.md
```

- Do **not** set `DECODEBENCH_F4_SPLITS` for validation runs (the byte model
  assumes the default one-tile-per-split; the script unsets it defensively).

## 3. What the collection produces

- `validation/results/timing.csv` — ≥30 trials × {f1,f2,f4} × {2048,4096} ×
  {unfused-stream, unfused-graph, fused}, batch=1 only.
- `validation/results/ncu_raw_{fusion}_{variant}_{dim}.csv` — 12 files; each
  must contain `dram__bytes_*` AND `gpu__time_duration.sum` rows (the
  duration metric feeds τ and the structural term S).
- `validation/results/ncu_metrics.csv` — per-dim bytes + `kernel_time_us`.
- `validation/results/validation_report.md` — the gated report.

## 4. Promote and commit

```bash
mkdir -p validation/results/<gpu>
cp -p validation/results/timing.csv validation/results/ncu_metrics.csv \
      validation/results/ncu_raw_*_2048.csv validation/results/ncu_raw_*_4096.csv \
      validation/results/validation_report.md validation/results/<gpu>/
```

Commit only the `results/<gpu>/` directory (top-level `results/*.csv|md` are
gitignored working outputs). Report the Overall verdict verbatim.

## 5. Expected outcomes (v2 predictions — see PREREGISTRATION-v2.md)

- Overall PASS with H1-v2 and H2-v2 holding.
- F4: fused < graph wall-clock; S > 0; S > B. Compare S/B against T4
  calibration (2.7 @2048, 4.7 @4096): predicted LARGER on GPUs with more SMs.
- F4 NCU byte-delta rows read "below resolution — no claim" (expected at
  these problem sizes on all planned GPUs).
- If any gate FAILs: commit the report as-is and record the refuted
  prediction. Do not tune tolerances on confirmatory data.
