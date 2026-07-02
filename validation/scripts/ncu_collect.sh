#!/usr/bin/env bash
# ncu_collect.sh — collect NCU metrics for DecodeBench validation
# Wraps ncu over F1/F2/F4 × {unfused-stream, fused} × dim {2048, 4096}, B=1.
# Output: ncu_raw_{fusion}_{variant}_{dim}.csv (one per combination), then
#         ncu_metrics.csv via analysis/parse_ncu.py.
set -euo pipefail

# compare.py models the default one-tile-per-split F4 configuration.
unset DECODEBENCH_F4_SPLITS

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

# gpu__time_duration.sum feeds the v2 structural term S: per-kernel isolated
# execution durations, summed per round (see analysis/parse_ncu.py and
# validation/PREREGISTRATION-v2.md).
NCU_METRICS="dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum"

run_ncu() {
  local fusion="$1"
  local variant="$2"
  local dim="$3"
  local batch=1
  local raw_csv="${OUTPUT_DIR}/ncu_raw_${fusion}_${variant}_${dim}.csv"

  echo "--- NCU: $fusion / $variant / dim=$dim ---"

  # Fail-closed: an NCU failure or empty output aborts collection. A missing
  # counter file must not silently flow downstream as "N/A".
  ncu \
    --metrics "$NCU_METRICS" \
    --clock-control none \
    --force-overwrite \
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
      --csv /dev/null

  if [ ! -s "$raw_csv" ]; then
    echo "ERROR: no NCU output for $fusion/$variant/dim=$dim ($raw_csv missing or empty)" >&2
    exit 1
  fi
}

for dim in 2048 4096; do
  # F1: RMSNorm→GEMV
  run_ncu f1 unfused-stream "$dim"
  run_ncu f1 fused "$dim"

  # F2: GEMV→SwiGLU
  run_ncu f2 unfused-stream "$dim"
  run_ncu f2 fused "$dim"

  # F4: FlashDecode attention
  run_ncu f4 unfused-stream "$dim"
  run_ncu f4 fused "$dim"
done

# Aggregate raw per-metric rows into ncu_metrics.csv (round-aware median)
python3 "${VAL_DIR}/analysis/parse_ncu.py" --results-dir "$OUTPUT_DIR"
