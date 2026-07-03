"""Cost model: per-operation byte formulas and predicted-bottleneck table.

Generates ``predicted.csv`` and a markdown summary table at a given
dimension and batch size.
"""
from __future__ import annotations
import csv
import io
import os
import sys
from dataclasses import dataclass
from typing import Literal

from decodebench.bytes_model import StageTrace, total_bytes, eliminable_bytes

_FFN = {2048: 8192, 4096: 11008}

def _f1_traces(d: int, b: int = 1) -> list[StageTrace]:
    """RMSNorm -> GEMV."""
    return [
        StageTrace("rmsnorm", reads=[2 * b * d, 2 * d], write=2 * b * d, is_final=False),
        StageTrace("gemv", reads=[2 * d * d, 2 * b * d], write=2 * b * d, is_final=True)
    ]

def _f2_traces(d: int, b: int = 1) -> list[StageTrace]:
    """Gate GEMV -> up GEMV -> SwiGLU.

    Both GEMV outputs are materialized by the unfused pipeline and are read by
    SwiGLU, so both write/read pairs are eliminable.
    """
    ff = _FFN[d]
    return [
        StageTrace("gate", reads=[2 * b * d, 2 * ff * d], write=2 * b * ff, is_final=False),
        StageTrace("up", reads=[2 * b * d, 2 * ff * d], write=2 * b * ff, is_final=False),
        StageTrace("swiglu", reads=[2 * b * ff, 2 * b * ff], write=2 * b * ff, is_final=True),
    ]

def _f4_traces(d: int, b: int = 1, l: int = 1024, head_dim: int = 128) -> list[StageTrace]:
    """FlashDecode: attention scores -> softmax -> weighted sum."""
    h = d // head_dim
    return [
        StageTrace("scores", reads=[2 * b * d, 2 * b * h * l * head_dim], write=4 * b * h * l, is_final=False),
        StageTrace("softmax", reads=[4 * b * h * l], write=4 * b * h * l, is_final=False),
        StageTrace("weighted_v", reads=[4 * b * h * l, 2 * b * h * l * head_dim], write=2 * b * d, is_final=True)
    ]

@dataclass(frozen=True)
class FusionCost:
    name: str
    total: int
    eliminable: int
    ratio: float
    predicted: Literal["low-byte-opportunity", "material-byte-opportunity"]

def compute_fusion_costs(
    d: int = 4096, b: int = 1, threshold: float = 0.01
) -> list[FusionCost]:
    """Return per-fusion byte totals and predicted bottleneck labels."""
    builders = {"F1": _f1_traces, "F2": _f2_traces, "F4": _f4_traces}
    out: list[FusionCost] = []
    for name, builder in builders.items():
        traces = builder(d, b)
        tot = total_bytes(traces)
        elim = eliminable_bytes(traces)
        ratio = elim / tot if tot else 0.0
        predicted: Literal["low-byte-opportunity", "material-byte-opportunity"] = (
            "material-byte-opportunity" if ratio >= threshold
            else "low-byte-opportunity"
        )
        out.append(FusionCost(name, tot, elim, ratio, predicted))
    return out

def to_predicted_csv(costs: list[FusionCost]) -> str:
    """``predicted.csv`` content."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["fusion", "total_bytes", "eliminable_bytes", "ratio", "predicted"])
    for c in costs:
        w.writerow([c.name, c.total, c.eliminable, f"{c.ratio:.6f}", c.predicted])
    return buf.getvalue()

def to_markdown_table(costs: list[FusionCost]) -> str:
    """GFM table of fusion byte costs."""
    header = (
        "| Fusion | Unfused traffic (MB) | Eliminable (B) | Ratio   | Predicted bound |"
    )
    sep = (
        "|--------|----------------------|----------------|---------|-----------------|"
    )
    lines = [header, sep]
    for c in costs:
        mb = c.total / 1_000_000
        pct = c.ratio * 100
        lines.append(
            f"| {c.name} | {mb:.1f} MB | {c.eliminable:,d} B | "
            f"{pct:.3f}% | {c.predicted} |"
        )
    return "\n".join(lines)

def main():
    costs = compute_fusion_costs(d=4096, b=1, threshold=0.01)
    csv_str = to_predicted_csv(costs)
    with open("predicted.csv", "w") as f:
        f.write(csv_str)
    print("Generated predicted.csv:")
    print(csv_str)
    print("Markdown Table:")
    print(to_markdown_table(costs))

if __name__ == "__main__":
    main()
