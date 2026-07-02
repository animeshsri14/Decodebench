#!/usr/bin/env bash
# ncu_collect.sh — collect NCU metrics for DecodeBench validation
# Wraps ncu over F1/F2/F4 × {unfused-stream, fused} at dim=4096, B=1.
# Output: ncu_raw_{fusion}_{variant}.csv (one per combination), then
#         ncu_metrics.csv via analysis/parse_ncu.py.
set -euo pipefail

if [ -d "/usr/lib/wsl/lib" ]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VAL_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${BUILD_DIR:-${VAL_DIR}/build}"
BENCH_BIN="${BUILD_DIR}/bench_variant"

if [ ! -x "$BENCH_BIN" ]; then
  echo "ERROR: bench_variant not found at $BENCH_BIN. Build first."
  exit 1
fi

OUTPUT_DIR="${VAL_DIR}/results"
mkdir -p "$OUTPUT_DIR"

NCU_METRICS="dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed"

run_ncu() {
  local fusion="$1"
  local variant="$2"
  local dim=4096
  local batch=1
  local raw_csv="${OUTPUT_DIR}/ncu_raw_${fusion}_${variant}.csv"

  echo "--- NCU: $fusion / $variant ---"

  ncu \
    --metrics "$NCU_METRICS" \
    --clock-control none \
    --csv \
    --log-file "$raw_csv" \
    "$BENCH_BIN" \
      --fusion "$fusion" \
      --variant "$variant" \
      --dim "$dim" \
      --batch "$batch" \
      --trials 1 \
      --target-ms 1 \
      --ncu-mode \
      --skip-correctness \
      --csv /dev/null \
    2>&1 || echo "WARN: ncu failed for $fusion/$variant"

  if [ ! -s "$raw_csv" ]; then
    echo "WARN: no NCU output for $fusion/$variant ($raw_csv missing or empty)"
  fi
}

# F1: RMSNorm→GEMV
run_ncu f1 unfused-stream
run_ncu f1 fused

# F2: GEMV→SwiGLU
run_ncu f2 unfused-stream
run_ncu f2 fused

# F4: FlashDecode attention
run_ncu f4 unfused-stream
run_ncu f4 fused

# Aggregate raw per-metric rows into ncu_metrics.csv (round-aware median)
python3 "${VAL_DIR}/analysis/parse_ncu.py" --results-dir "$OUTPUT_DIR"
