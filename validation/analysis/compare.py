#!/usr/bin/env python3
# compare.py — DecodeBench validation analysis (fail-closed).
# Joins timing CSV + ncu CSV, computes byte models, emits validation_report.md.
#
# Design rules (2026-07 revision, after external review):
#   * Fail-closed: every expected (fusion, dim, batch, variant) config MUST be
#     present with usable samples; every expected NCU cell MUST have data.
#     Missing data is a FAIL, never a silent skip or "N/A".
#   * The overall verdict is computed from structured check records, never by
#     string-matching rendered report lines.
#   * Medians (not means) summarize timing samples, matching the Python
#     library's robust statistics.
#   * Pre-registered hypotheses (README §Pre-registered Hypotheses) are
#     enforced as gates: H1/H2 (≥80% attribution), H4(c) (decomposition
#     within 2% of t_graph). WARN never counts as PASS.
#
# Checks:
#   (G0) data completeness gate
#   (G1) numerical correctness gate from bench_variant
#   (a)  residual_us = t_graph - t_fused - B within +2%·t_graph
#   (b)  analytic vs NCU DRAM bytes: F1/F2 absolute ±20%;
#        F4 two-sided eliminated-delta gate vs the analytic delta
#        (eliminable bytes minus modeled split-KV partial traffic)
#   (c)  Δ_launch = t_stream - t_graph ≥ 0
#   (H)  pre-registered hypotheses H1, H2, H4(c)

import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime

from decodebench.bytes_model import StageTrace, eliminable_bytes, total_bytes

# ---- Byte model matching bench_variant.cu actual dimensions ----
#
# F1: bench_variant uses d_in=dim, d_out=14336 (hardcoded)
# F2: bench_variant uses d_in=dim, d_out=14336 (hardcoded)
# F4: bench_variant uses H=32, D=128, L=dim (dim is the KV-cache length)
#
# All values in bytes, FP16 = 2 bytes per element.
_D_OUT_F1F2 = 14336
_N_HEADS = 32
_HEAD_DIM = 128
_F4_TILE_L = 128  # default split-KV: one tile per split (bench_variant default)

FUSIONS = ["f1", "f2", "f4"]
DIMS = ["2048", "4096"]
BATCHES = ["1"]  # kernels are unbatched; bench_variant rejects batch != 1
VARIANTS = ["unfused-stream", "unfused-graph", "fused"]


def _traces_f1(dim):
    d = dim
    d_out = _D_OUT_F1F2
    return [
        StageTrace("rmsnorm", reads=[d * 2, d * 2], write=d * 2, is_final=False),
        StageTrace("gemv", reads=[d * 2, d_out * d * 2], write=d_out * 2, is_final=True),
    ]


def _traces_f2(dim):
    """gate GEMV -> up GEMV -> SwiGLU: BOTH intermediates (g and u) are
    materialized by the unfused pipeline and both are eliminable. The earlier
    two-stage model hid u inside a combined up+swiglu stage and understated
    total and eliminable bytes."""
    d = dim
    d_out = _D_OUT_F1F2
    return [
        StageTrace("gate", reads=[d * 2, d_out * d * 2], write=d_out * 2, is_final=False),
        StageTrace("up", reads=[d * 2, d_out * d * 2], write=d_out * 2, is_final=False),
        StageTrace("swiglu", reads=[d_out * 2, d_out * 2], write=d_out * 2, is_final=True),
    ]


def _traces_f4(dim):
    """attention scores -> softmax -> weighted V (dim = KV length L)."""
    H, D, L = _N_HEADS, _HEAD_DIM, dim
    return [
        StageTrace("scores", reads=[H * D * 2, H * L * D * 2], write=H * L * 4, is_final=False),
        StageTrace("softmax", reads=[H * L * 4], write=H * L * 4, is_final=False),
        StageTrace(
            "weighted_v", reads=[H * L * 4, H * L * D * 2], write=H * D * 2, is_final=True
        ),
    ]


_TRACE_BUILDERS = {"f1": _traces_f1, "f2": _traces_f2, "f4": _traces_f4}


def f4_fused_partial_bytes(dim):
    """Global traffic the split-KV fused F4 ADDS relative to a fully fused
    kernel: per-split partial output/max/sum buffers, each written by
    f4_partial_kernel and read by f4_reduce_kernel (write + read = x2).
    Assumes the bench default n_splits = L / TILE_L; if DECODEBENCH_F4_SPLITS
    was overridden during collection this model does not apply.
    """
    H, D, L = _N_HEADS, _HEAD_DIM, dim
    n_splits = L // _F4_TILE_L
    per_split = D * 4 + 4 + 4  # part_o row (fp32) + part_m + part_l
    return 2 * H * n_splits * per_split


def compute_B(fusion, dim, t_graph):
    """B = eliminable_bytes / achieved_bw, achieved_bw = total_bytes / t_graph.
    NOTE: proportional byte-time estimate, not an upper bound (see
    decodebench.verdict docstring)."""
    if t_graph <= 0:
        return 0.0
    builder = _TRACE_BUILDERS.get(fusion)
    if builder is None:
        return 0.0
    traces = builder(dim)
    total = total_bytes(traces)
    elim = eliminable_bytes(traces)
    achieved_bw = total / t_graph
    return elim / achieved_bw if achieved_bw > 0 else 0.0


def load_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


class Checks:
    """Structured check registry; the overall verdict comes from here."""

    def __init__(self):
        self.records = []  # dicts: {section, label, status, detail}

    def add(self, section, label, status, detail=""):
        assert status in ("PASS", "FAIL", "WARN")
        self.records.append(
            {"section": section, "label": label, "status": status, "detail": detail}
        )
        return status

    def counts(self):
        c = {"PASS": 0, "FAIL": 0, "WARN": 0}
        for r in self.records:
            c[r["status"]] += 1
        return c

    def overall(self):
        # Fail-closed: WARN means incomplete/indeterminate, never a valid PASS.
        c = self.counts()
        if c["FAIL"] > 0:
            return "FAIL"
        return "INCOMPLETE" if c["WARN"] > 0 else "PASS"


def residual_matches_model(t_graph, t_fused, byte_est, relative_tol=0.02):
    residual = t_graph - t_fused - byte_est
    return abs(residual) <= relative_tol * t_graph, residual


def relative_delta_matches(measured, analytic, relative_tol):
    if analytic <= 0 or relative_tol < 0:
        return False
    return abs(measured - analytic) <= relative_tol * analytic


def main():
    parser = argparse.ArgumentParser(description="DecodeBench validation analysis")
    parser.add_argument("--timing-csv", required=True)
    parser.add_argument("--ncu-csv", required=True)
    parser.add_argument("--output", default="validation_report.md")
    parser.add_argument(
        "--f4-delta-tol", type=float, default=0.5,
        help="Two-sided relative tolerance for the F4 eliminated-delta gate "
             "(measured delta must lie within analytic_delta*(1±tol)).",
    )
    parser.add_argument(
        "--min-timing-samples", type=int, default=30,
        help="Minimum usable trials required for every timing cell (default: 30).",
    )
    parser.add_argument(
        "--allow-missing-ncu", action="store_true",
        help="Downgrade missing NCU data from FAIL to WARN (dev runs only; "
             "a report produced this way is not a valid validation).",
    )
    args = parser.parse_args()
    if not 0 <= args.f4_delta_tol < 1:
        parser.error("--f4-delta-tol must be in [0, 1)")
    if args.min_timing_samples <= 0:
        parser.error("--min-timing-samples must be positive")

    timing = load_csv(args.timing_csv)
    ncu_data = load_csv(args.ncu_csv) if os.path.exists(args.ncu_csv) else []

    ncu_index = {
        (row.get("fusion", ""), row.get("variant", "")): row for row in ncu_data
    }

    # Group timing by (fusion, dim, batch, variant). Batch is part of the key:
    # pooling nominal batches as repetitions fabricated batch data before.
    groups = defaultdict(list)
    for row in timing:
        key = (row["fusion"], row["dim"], row.get("batch", "1"), row["variant"])
        groups[key].append(row)

    checks = Checks()
    report = []
    report.append("# DecodeBench Validation Report")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")

    # ---- Completeness gate (fail-closed) ----
    report.append("## Check (G0): Data completeness")
    report.append("")
    report.append(
        "Every expected (fusion, dim, batch, variant) config must be present with "
        "usable timing samples, and every expected NCU cell must have data. A "
        "missing cell is a FAIL: an empty or partial collection must not be able "
        "to produce an overall PASS."
    )
    report.append("")
    medians = {}       # (fusion, dim, batch, variant) -> median us
    correctness = {}   # same key -> 0/1
    for fusion in FUSIONS:
        for dim in DIMS:
            for batch in BATCHES:
                for variant in VARIANTS:
                    key = (fusion, dim, batch, variant)
                    rows = groups.get(key, [])
                    us_vals = []
                    for r in rows:
                        try:
                            value = float(r.get("us_per_invocation", 0) or 0)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(value) and value > 0.001:
                            us_vals.append(value)
                    label = f"{fusion}/dim={dim}/b={batch}/{variant}"
                    if len(us_vals) < args.min_timing_samples:
                        detail = (f"{len(us_vals)} usable timing samples; "
                                  f"require >= {args.min_timing_samples}")
                        checks.add("G0", label, "FAIL", detail)
                        report.append(f"- {label}: **FAIL** ({detail})")
                        continue
                    medians[key] = statistics.median(us_vals)
                    oks = []
                    for r in rows:
                        try:
                            oks.append(int(r.get("correctness_ok", 0) or 0))
                        except (TypeError, ValueError):
                            oks.append(0)
                    correctness[key] = 1 if oks and all(v == 1 for v in oks) else 0
                    checks.add("G0", label, "PASS", f"n={len(us_vals)}")
    for fusion in FUSIONS:
        for variant in ["unfused-stream", "fused"]:
            row = ncu_index.get((fusion, variant))
            try:
                ncu_total = (float(row.get("dram_bytes_read", 0) or 0)
                             + float(row.get("dram_bytes_write", 0) or 0)) if row else 0
            except (TypeError, ValueError):
                ncu_total = 0
            has = math.isfinite(ncu_total) and ncu_total > 0
            label = f"ncu {fusion}/{variant}"
            if has:
                checks.add("G0", label, "PASS")
            else:
                status = "WARN" if args.allow_missing_ncu else "FAIL"
                checks.add("G0", label, status, "no NCU data")
                report.append(f"- {label}: **{status}** (no NCU data)")
    report.append("- (unlisted cells present and usable)")
    report.append("")

    def med(fusion, dim, variant, batch="1"):
        return medians.get((fusion, dim, batch, variant), 0.0)

    # ---- G1 correctness ----
    report.append("## Check (G1): Numerical correctness (fused/unfused vs CPU reference)")
    report.append("")
    report.append("| Fusion | Dim | Variant | correctness_ok | Status |")
    report.append("|--------|-----|---------|----------------|--------|")
    for fusion in FUSIONS:
        for dim in DIMS:
            for variant in VARIANTS:
                key = (fusion, dim, "1", variant)
                if key not in correctness:
                    continue  # already FAILed in G0
                ok = correctness[key]
                status = checks.add(
                    "G1", f"{fusion}/{dim}/{variant}", "PASS" if ok else "FAIL"
                )
                report.append(f"| {fusion} | {dim} | {variant} | {ok} | {status} |")
    report.append("")

    # ---- Check (a): residual ----
    report.append("## Check (a): Residual analysis (t_graph - t_fused - B)")
    report.append("")
    report.append(
        "PASS means the fused speedup over the graph baseline is explained by the "
        "byte estimate B within +2% of t_graph. A larger residual — in either "
        "the unfavorable OR the favorable direction — is a FAIL: a fused win far "
        "beyond Δ_launch + B means the decomposition does not describe this "
        "workload, whatever the sign of the surprise. [2026-07 revision: the "
        "earlier favorable-direction WARN reclassification is reverted.]"
    )
    report.append("")
    report.append("| Fusion | Dim | t_graph (us) | t_fused (us) | B (us) | Residual | Status |")
    report.append("|--------|-----|-------------|-------------|--------|----------|--------|")
    for fusion in FUSIONS:
        for dim in DIMS:
            t_graph = med(fusion, dim, "unfused-graph")
            t_fused = med(fusion, dim, "fused")
            if not (t_graph > 0 and t_fused > 0):
                continue  # missing data already FAILed in G0
            B = compute_B(fusion, int(dim), t_graph)
            ok, residual = residual_matches_model(t_graph, t_fused, B)
            status = checks.add(
                "a", f"{fusion}/{dim}", "PASS" if ok else "FAIL",
                f"residual={residual:.2f}us",
            )
            report.append(
                f"| {fusion} | {dim} | {t_graph:.2f} | {t_fused:.2f} | {B:.2f} "
                f"| {residual:.2f} | {status} |"
            )
    report.append("")

    # ---- Check (b): analytic vs NCU DRAM bytes ----
    report.append("## Check (b): Analytic bytes vs NCU DRAM bytes")
    report.append("")
    report.append(
        "F1/F2 gate on absolute totals (tolerance ±20%; analytic is a lower bound; "
        "cache-line granularity and L2 thrashing add measured ~10-15% on weight "
        "streams). F4 gates on the eliminated DELTA, two-sided: measured "
        "(unfused − fused) DRAM bytes must lie within ±{:.0f}% of the analytic "
        "delta (eliminable bytes minus the modeled split-KV partial-buffer "
        "traffic the fused variant adds). A delta far ABOVE the model is as much "
        "a model failure as one below it — an unbounded one-sided gate would "
        "pass on any unrelated traffic difference.".format(args.f4_delta_tol * 100)
    )
    report.append("")
    report.append("| Fusion | Variant | Analytic (MB) | NCU DRAM (MB) | Ratio | Status |")
    report.append("|--------|---------|---------------|---------------|-------|--------|")

    def _ncu_total_mb(fusion, variant):
        ncu = ncu_index.get((fusion, variant), {})
        try:
            read = float(ncu.get("dram_bytes_read", 0) or 0)
            write = float(ncu.get("dram_bytes_write", 0) or 0)
        except ValueError:
            return 0.0
        return (read + write) / 1e6

    for fusion in FUSIONS:
        builder = _TRACE_BUILDERS[fusion]
        analytic_mb = total_bytes(builder(4096)) / 1e6
        elim_mb = eliminable_bytes(builder(4096)) / 1e6
        totals = {v: _ncu_total_mb(fusion, v) for v in ["unfused-stream", "fused"]}
        delta_mb = totals["unfused-stream"] - totals["fused"]

        if fusion == "f4":
            if totals["unfused-stream"] > 0 and totals["fused"] > 0:
                analytic_delta_mb = elim_mb - f4_fused_partial_bytes(4096) / 1e6
                lo = analytic_delta_mb * (1 - args.f4_delta_tol)
                hi = analytic_delta_mb * (1 + args.f4_delta_tol)
                ok = relative_delta_matches(delta_mb, analytic_delta_mb, args.f4_delta_tol)
                status = checks.add(
                    "b", "f4/delta", "PASS" if ok else "FAIL",
                    f"measured delta {delta_mb:.2f} MB vs analytic "
                    f"{analytic_delta_mb:.2f} MB (accept [{lo:.2f}, {hi:.2f}])",
                )
                report.append(
                    f"| f4 | delta (unfused−fused) | {analytic_delta_mb:.2f} | "
                    f"{delta_mb:.2f} | "
                    f"{delta_mb / analytic_delta_mb if analytic_delta_mb else 0:.2f} "
                    f"| {status} |"
                )
            # absolute totals stay as printed diagnostics (no gate)
            for variant in ["unfused-stream", "fused"]:
                ncu_mb = totals[variant]
                ratio = analytic_mb / ncu_mb if ncu_mb > 0 else 0
                report.append(
                    f"| f4 | {variant} (diagnostic) | {analytic_mb:.2f} | "
                    f"{ncu_mb:.2f} | {ratio:.2f} | — |"
                )
        else:
            for variant in ["unfused-stream", "fused"]:
                ncu_mb = totals[variant]
                if ncu_mb <= 0:
                    continue  # missing NCU already handled in G0
                ratio = analytic_mb / ncu_mb
                status = checks.add(
                    "b", f"{fusion}/{variant}",
                    "PASS" if abs(ratio - 1.0) <= 0.20 else "FAIL",
                    f"ratio={ratio:.2f}",
                )
                report.append(
                    f"| {fusion} | {variant} | {analytic_mb:.2f} | {ncu_mb:.2f} "
                    f"| {ratio:.2f} | {status} |"
                )
    report.append("")

    # ---- Check (c): Δ_launch ----
    report.append("## Check (c): CUDA Graphs capture launch overhead (Δ_launch ≥ 0)")
    report.append("")
    report.append("| Fusion | Dim | t_stream (us) | t_graph (us) | Δ_launch (us) | Status |")
    report.append("|--------|-----|--------------|-------------|--------------|--------|")
    for fusion in FUSIONS:
        for dim in DIMS:
            t_stream = med(fusion, dim, "unfused-stream")
            t_graph = med(fusion, dim, "unfused-graph")
            if not (t_stream > 0 and t_graph > 0):
                continue
            delta_launch = t_stream - t_graph
            status = checks.add(
                "c", f"{fusion}/{dim}", "PASS" if delta_launch >= 0 else "FAIL",
                f"delta={delta_launch:.2f}us",
            )
            report.append(
                f"| {fusion} | {dim} | {t_stream:.2f} | {t_graph:.2f} "
                f"| {delta_launch:.2f} | {status} |"
            )
    report.append("")

    # ---- Check (H): pre-registered hypotheses ----
    report.append("## Check (H): Pre-registered hypotheses (README)")
    report.append("")
    report.append(
        "H1: F1/F2 launch-bound — Δ_launch explains ≥80% of the unfused-to-fused "
        "gap (t_stream − t_fused). H2: F4 byte-bound — B explains ≥80% of that "
        "gap. H4(c): |(t_stream − t_fused) − (Δ_launch + B)| ≤ 2% of t_graph. "
        "These are gates, not notes: a refuted hypothesis is a FAIL in this "
        "report (and belongs in the paper as a negative result). When the fused "
        "kernel provides no gain over the stream baseline (gap ≤ 0) the "
        "attribution fraction is undefined; that is reported as WARN with the "
        "gap shown, since the pre-registered claim is about a positive gap."
    )
    report.append("")
    report.append("| Hypothesis | Fusion | Dim | Gap (us) | Term (us) | Fraction | Status |")
    report.append("|-----------|--------|-----|----------|-----------|----------|--------|")
    for fusion in FUSIONS:
        hyp = "H2" if fusion == "f4" else "H1"
        for dim in DIMS:
            t_stream = med(fusion, dim, "unfused-stream")
            t_graph = med(fusion, dim, "unfused-graph")
            t_fused = med(fusion, dim, "fused")
            if not (t_stream > 0 and t_graph > 0 and t_fused > 0):
                continue
            gap = t_stream - t_fused
            delta_launch = t_stream - t_graph
            B = compute_B(fusion, int(dim), t_graph)
            term = B if fusion == "f4" else delta_launch
            if gap <= 0:
                status = checks.add(
                    "H", f"{hyp}/{fusion}/{dim}", "WARN",
                    f"gap={gap:.2f}us <= 0; attribution fraction undefined",
                )
                report.append(
                    f"| {hyp} | {fusion} | {dim} | {gap:.2f} | {term:.2f} | — | {status} |"
                )
            else:
                frac = term / gap
                status = checks.add(
                    "H", f"{hyp}/{fusion}/{dim}", "PASS" if frac >= 0.8 else "FAIL",
                    f"fraction={frac:.2f}",
                )
                report.append(
                    f"| {hyp} | {fusion} | {dim} | {gap:.2f} | {term:.2f} "
                    f"| {frac:.2f} | {status} |"
                )
            # H4(c) decomposition consistency
            resid = abs(gap - (delta_launch + B))
            status = checks.add(
                "H", f"H4c/{fusion}/{dim}",
                "PASS" if resid <= 0.02 * t_graph else "FAIL",
                f"|gap-(Δ+B)|={resid:.2f}us vs 2%·t_graph={0.02 * t_graph:.2f}us",
            )
            report.append(
                f"| H4(c) | {fusion} | {dim} | {gap:.2f} | "
                f"{delta_launch + B:.2f} | — | {status} |"
            )
    report.append("")

    # ---- Summary (structured, never string-matched) ----
    c = checks.counts()
    overall = checks.overall()
    report.append("---")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append(f"- PASS: {c['PASS']}")
    report.append(f"- WARN: {c['WARN']} (warnings never count as passes)")
    report.append(f"- FAIL: {c['FAIL']}")
    report.append(f"- **Overall: {overall}**")
    if c["FAIL"] > 0:
        report.append("")
        report.append("### Failing checks")
        for r in checks.records:
            if r["status"] == "FAIL":
                report.append(f"- [{r['section']}] {r['label']}: {r['detail']}")

    report_text = "\n".join(report)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport written to {args.output}")
    # Non-zero exit on FAIL so the shell pipeline cannot fail open.
    sys.exit(0 if c["FAIL"] == 0 and c["WARN"] == 0 else 1)


if __name__ == "__main__":
    main()
