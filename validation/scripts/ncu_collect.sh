#!/usr/bin/env bash
# ncu_collect.sh — collect NCU metrics for DecodeBench validation
# Wraps ncu over F1/F2/F4 × {unfused-stream, fused} at dim=4096, B=1.
# Output: ncu_metrics.csv
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
NCU_CSV="${OUTPUT_DIR}/ncu_metrics.csv"

echo "fusion,variant,dram_bytes_read,dram_bytes_write,l2_hit_rate_pct,dram_throughput_pct" > "$NCU_CSV"

run_ncu() {
  local fusion="$1"
  local variant="$2"
  local dim=4096
  local batch=1
  local tmp_csv="${OUTPUT_DIR}/ncu_tmp_${fusion}_${variant}.csv"

  echo "--- NCU: $fusion / $variant ---"

  ncu \
    --metrics "$NCU_METRICS" \
    --clock-control none \
    --csv \
    --log-file "$tmp_csv" \
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

  # Extract the metric line (second line after header)
  if [ -f "$tmp_csv" ]; then
    local vals
    vals=$(tail -n +2 "$tmp_csv" | head -1)
    if [ -n "$vals" ]; then
      # Parse comma-separated metric values
      local dram_read=$(echo "$vals" | cut -d',' -f1)
      local dram_write=$(echo "$vals" | cut -d',' -f2)
      local l2_hit=$(echo "$vals" | cut -d',' -f3)
      local dram_pct=$(echo "$vals" | cut -d',' -f4)
      echo "${fusion},${variant},${dram_read},${dram_write},${l2_hit},${dram_pct}" >> "$NCU_CSV"
    else
      echo "${fusion},${variant},NA,NA,NA,NA" >> "$NCU_CSV"
    fi
  else
    echo "${fusion},${variant},NA,NA,NA,NA" >> "$NCU_CSV"
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

echo "NCU metrics written to $NCU_CSV"
