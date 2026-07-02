#!/usr/bin/env python3
# parse_ncu.py — Convert per-metric-row NCU raw CSVs into a clean ncu_metrics.csv
#
# NCU raw CSV format (from ncu --csv --metrics ...):
#   One row per metric per kernel invocation.
#   ID column increments globally across all kernel launches in the process.
#   Multiple kernels per "round" (e.g., unfused F4 = scores+softmax+attn_v per round).
#
# This script:
#   1. Reads ncu_raw_{fusion}_{variant}.csv for all 6 combinations.
#   2. Detects round size (period of repeating kernel-name sequence).
#   3. Sums dram__bytes_read.sum + dram__bytes_write.sum across all kernels in a round.
#   4. Takes the median across rounds (skips first round as warmup).
#   5. Emits results/ncu_metrics.csv consumed by compare.py --ncu-csv.

import argparse
import csv
import os
import statistics

FUSIONS = ["f1", "f2", "f4"]
VARIANTS = ["unfused-stream", "fused"]


def find_round_size(kernel_sequence):
    """Return the period of the repeating kernel-name sequence.

    Example: [rmsnorm, gemv, rmsnorm, gemv] -> 2
             [gemv, gemv, swiglu, gemv, gemv, swiglu] -> 3
             [f1_kernel, f1_kernel, ...] -> 1
    """
    names = [kname for _, kname in kernel_sequence]
    n = len(names)
    # The candidate period must fit the ENTIRE sequence. Profiler-injected
    # kernels, renamed templates, or truncated captures break periodicity;
    # that raises instead of silently grouping counters into wrong rounds.
    for period in range(1, n // 2 + 1):
        pattern = names[:period]
        if all(names[i] == pattern[i % period] for i in range(n)):
            return period
    if len(set(names)) == 1:
        return 1
    raise ValueError(
        "NCU kernel sequence is not periodic; cannot group rounds reliably. "
        f"Distinct kernels seen: {sorted(set(names))}. First 12: {names[:12]}. "
        "Check for profiler-injected or unexpected kernels in the capture."
    )


def parse_ncu_file(path):
    """Parse one ncu_raw file, return (median_read_bytes, median_write_bytes)."""
    if not os.path.exists(path):
        return None, None

    with open(path, newline="", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("==")]

    reader = csv.DictReader(lines)
    rows = list(reader)
    if not rows:
        return None, None

    # Collect per-invocation DRAM bytes
    by_id = {}
    kernel_sequence = []  # ordered list of (id, kernel_name)

    for row in rows:
        inv_id = int(row["ID"])
        metric = row["Metric Name"].strip()
        try:
            val = float(row["Metric Value"].replace(",", "").strip())
        except (ValueError, AttributeError):
            continue
        kname = row["Kernel Name"].strip()

        if inv_id not in by_id:
            by_id[inv_id] = {"kernel": kname, "read": 0.0, "write": 0.0}
            kernel_sequence.append((inv_id, kname))

        if "dram__bytes_read" in metric:
            by_id[inv_id]["read"] += val
        elif "dram__bytes_write" in metric:
            by_id[inv_id]["write"] += val

    if not by_id:
        return None, None

    round_size = find_round_size(kernel_sequence)
    sorted_ids = sorted(by_id.keys())

    # Build per-round totals
    reads, writes = [], []
    for r_start in range(0, len(sorted_ids), round_size):
        chunk = sorted_ids[r_start : r_start + round_size]
        if len(chunk) < round_size:
            break  # incomplete trailing round
        reads.append(sum(by_id[i]["read"] for i in chunk))
        writes.append(sum(by_id[i]["write"] for i in chunk))

    if not reads:
        return None, None

    # Skip first round (warmup), take median of the rest; fall back to all if only 1 round
    if len(reads) > 1:
        reads = reads[1:]
        writes = writes[1:]

    return statistics.median(reads), statistics.median(writes)


def main():
    default_results = os.path.join(os.path.dirname(__file__), "..", "results")
    parser = argparse.ArgumentParser(description="Parse NCU raw CSVs into ncu_metrics.csv")
    parser.add_argument(
        "--results-dir",
        default=default_results,
        help="Directory containing ncu_raw_*.csv files and where ncu_metrics.csv is written "
             "(default: ../results relative to this script)",
    )
    args = parser.parse_args()
    results_dir = os.path.abspath(args.results_dir)
    output_csv = os.path.join(results_dir, "ncu_metrics.csv")

    rows = []
    for fusion in FUSIONS:
        for variant in VARIANTS:
            fname = f"ncu_raw_{fusion}_{variant}.csv"
            path = os.path.join(results_dir, fname)
            read_bytes, write_bytes = parse_ncu_file(path)

            if read_bytes is None:
                # Emit a zero row: compare.py's completeness gate (G0) treats
                # zero-byte cells as missing data and FAILs the run.
                print(f"  WARNING: no data for {fusion}/{variant} ({fname}); "
                      "compare.py will FAIL this cell")
                read_bytes, write_bytes = 0.0, 0.0
            else:
                total_mb = (read_bytes + write_bytes) / 1e6
                print(
                    f"  {fusion}/{variant}: read={read_bytes/1e6:.2f} MB "
                    f"write={write_bytes/1e6:.2f} MB total={total_mb:.2f} MB"
                )

            rows.append(
                {
                    "fusion": fusion,
                    "variant": variant,
                    "dram_bytes_read": int(read_bytes),
                    "dram_bytes_write": int(write_bytes),
                }
            )

    os.makedirs(results_dir, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["fusion", "variant", "dram_bytes_read", "dram_bytes_write"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {output_csv}")


if __name__ == "__main__":
    main()
