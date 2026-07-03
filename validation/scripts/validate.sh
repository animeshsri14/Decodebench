#!/usr/bin/env bash
# validate.sh — full DecodeBench validation pipeline
# Orchestrates: env check → G2 calibration → bench_variant grid →
#               ncu_collect → compare.py → validation report
set -euo pipefail

# The official analytic F4 delta assumes the default one-tile-per-split
# configuration. Do not let a caller's tuning override silently invalidate it.
unset DECODEBENCH_F4_SPLITS

if [ -d "/usr/lib/wsl/lib" ]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VAL_DIR="$(dirname "$SCRIPT_DIR")"
PROJ_ROOT="$(dirname "$VAL_DIR")"
BUILD_DIR="${BUILD_DIR:-${VAL_DIR}/build}"
RESULTS_DIR="${VAL_DIR}/results"
TIMING_CSV="${RESULTS_DIR}/timing.csv"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " DecodeBench Validation Pipeline"
echo "============================================"
echo "Project root: $PROJ_ROOT"
echo "Results dir:  $RESULTS_DIR"
echo

# ---- Step 1: Environment check ----
echo "=== Step 1: check_env.sh ==="
bash "${SCRIPT_DIR}/check_env.sh"

# ---- Step 2: G2 calibration ----
echo "=== Step 2: G2 calibration (gate-g2) ==="
CALIBRATE_BIN="${BUILD_DIR}/calibrate"
if [ -x "$CALIBRATE_BIN" ]; then
  "$CALIBRATE_BIN" --gate-g2
else
  echo "ERROR: calibrate binary not found at $CALIBRATE_BIN" >&2
  exit 1
fi

BENCH_BIN="${BUILD_DIR}/bench_variant"
if [ ! -x "$BENCH_BIN" ]; then
  echo "ERROR: bench_variant not found at $BENCH_BIN. Build first."
  exit 1
fi

# ---- Step 3: bench_variant over grid ----
echo "=== Step 3: Timing grid ==="
echo "gpu_name,fusion,variant,dim,batch,trial,iters,us_per_invocation,correctness_ok,timestamp" > "$TIMING_CSV"

# batch is fixed at 1: the kernels are unbatched and bench_variant rejects
# any other value (a batch sweep here previously produced fictitious data).
FUSIONS="f1 f2 f4"
VARIANTS=(unfused-stream unfused-graph fused)
DIMS="2048 4096"

# Best-effort clock locking: lock SM clocks to the reported maximum so slow
# clock drift cannot bias late-running variants. Fails harmlessly (recorded
# as a WARNING) on vGPU guests and hosts without permission — the interleaved
# passes below are the drift defense that always applies.
CLOCKS_LOCKED=0
MAX_SM_CLOCK="$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)"
if [ -n "${MAX_SM_CLOCK}" ] && [ "${MAX_SM_CLOCK}" != "[N/A]" ] \
   && sudo -n nvidia-smi -lgc "${MAX_SM_CLOCK},${MAX_SM_CLOCK}" >/dev/null 2>&1; then
  CLOCKS_LOCKED=1
  echo "GPU SM clocks locked at ${MAX_SM_CLOCK} MHz"
  trap 'sudo -n nvidia-smi -rgc >/dev/null 2>&1 || true' EXIT
else
  echo "WARNING: could not lock GPU clocks (vGPU guest or no passwordless sudo);"
  echo "         timings run at default clocks. Recorded as a run deviation."
fi

# Variants are measured in NPASSES rotated interleaved passes (same 30 total
# trials per cell as before) so thermal/clock drift spreads across variants
# instead of systematically biasing whichever variant ran last.
NPASSES=3
TRIALS_PER_PASS=10
NVAR=${#VARIANTS[@]}
for ((pass = 0; pass < NPASSES; pass++)); do
  for fusion in $FUSIONS; do
    for dim in $DIMS; do
      for ((vi = 0; vi < NVAR; vi++)); do
        variant="${VARIANTS[$(((vi + pass) % NVAR))]}"
        echo "  [pass $((pass + 1))/${NPASSES}] bench_variant --fusion $fusion --variant $variant --dim $dim"
        # A failed benchmark aborts the pipeline (fail-closed): a partial grid
        # must not flow into compare.py as if it were a complete run.
        "$BENCH_BIN" \
          --fusion "$fusion" \
          --variant "$variant" \
          --dim "$dim" \
          --batch 1 \
          --trials "${TRIALS_PER_PASS}" \
          --target-ms 20 \
          --seed 42 \
          --csv /dev/stdout | tail -n +2 >> "$TIMING_CSV"
      done
    done
  done
done

echo "Timing data written to $TIMING_CSV"

# ---- Step 4: NCU collection ----
echo "=== Step 4: ncu_collect.sh ==="
bash "${SCRIPT_DIR}/ncu_collect.sh"

# ---- Step 5: Analysis ----
echo "=== Step 5: compare.py ==="
REPORT="${RESULTS_DIR}/validation_report.md"
COMPARE_STATUS=0
# compare.py exits non-zero when any check FAILs (including missing data).
# That status is this pipeline's exit code — no downgrade to a warning.
python3 "${SCRIPT_DIR}/../analysis/compare.py" \
  --timing-csv "$TIMING_CSV" \
  --ncu-csv "${RESULTS_DIR}/ncu_metrics.csv" \
  --output "$REPORT" \
  2>&1 || COMPARE_STATUS=$?

echo
echo "============================================"
if [ "$COMPARE_STATUS" -ne 0 ]; then
  echo " Validation FAILED (see report)"
else
  echo " Validation complete"
fi
echo " Report: $REPORT"
echo "============================================"

exit "$COMPARE_STATUS"
