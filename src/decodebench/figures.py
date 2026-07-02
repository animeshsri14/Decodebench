"""Matplotlib figures - Agg backend set before any pyplot import."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend before any pyplot import

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from decodebench.verdict import Verdict

def plot_verdict_bar(verdicts: Dict[str, Verdict], path: str) -> None:
    """Grouped bars per demo: delta_launch (measured) vs B (byte estimate), in microseconds."""
    demos = list(verdicts.keys())
    if not demos:
        return

    delta_launches = [verdicts[name].delta_launch for name in demos]
    b_ests = [verdicts[name].b_bytes_est for name in demos]

    x = np.arange(len(demos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, delta_launches, width, label='Δ_launch (measured)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, b_ests, width, label='B (byte estimate)', color='#ff7f0e')

    ax.set_ylabel('Time (µs)')
    ax.set_title('What CUDA Graphs already buy vs. the most hand-fusion could add')
    ax.set_xticks(x)
    ax.set_xticklabels(demos)
    ax.legend()

    # Add values on top of bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f} µs',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_predicted_vs_measured(data: dict, path: str) -> None:
    """Figure 3: predicted vs measured (analytic B vs measured t_graph - t_fused).

    data: {fusion_name: (analytic_b_us, measured_us)}.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="gray")
    for name, (analytic_b, measured) in data.items():
        ax.scatter(analytic_b, measured, label=name)
        ax.annotate(name, (analytic_b, measured), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Analytic B (µs)")
    ax.set_ylabel("Measured t_graph - t_fused (µs)")
    ax.set_title("Analytic Byte Estimate vs. Hardware Gain")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_cross_arch(data: dict, path: str) -> None:
    """Figure 4: M and B (µs) per fusion, scatter vs GPU ridge point.

    data: {gpu_name: {"ridge": flop_per_byte,
                      "m_us": {fusion: us}, "b_us": {fusion: us}}}.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for gpu, entry in data.items():
        ridge = entry["ridge"]
        for fusion, m in entry.get("m_us", {}).items():
            ax.scatter(ridge, m, marker="o", label=f"{gpu} {fusion} Δ_launch")
        for fusion, b in entry.get("b_us", {}).items():
            ax.scatter(ridge, b, marker="^", label=f"{gpu} {fusion} B")
    if data:
        ax.legend(fontsize=7)
    ax.set_xlabel("GPU Ridge Point (FLOP/byte)")
    ax.set_ylabel("Time (µs)")
    ax.set_title("Cross-Architecture Performance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_batch_sweep(data: list[dict], path: str) -> None:
    """Figure 5: Δ_launch and B vs batch size.

    data: sweep summary CSV rows (§8.9 schema) — dicts with keys
    demo, dim, batch, delta_launch_us, b_bytes_est_us (older CSVs: b_ceiling_us).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    series: Dict[tuple, list] = {}
    for row in data:
        key = (row["demo"], row["dim"])
        b_us = row.get("b_bytes_est_us", row.get("b_ceiling_us"))
        series.setdefault(key, []).append(
            (int(row["batch"]), float(row["delta_launch_us"]), float(b_us))
        )
    for (demo, dim), points in series.items():
        points.sort()
        batches = [p[0] for p in points]
        ax.plot(batches, [p[1] for p in points], marker="o",
                label=f"{demo} dim={dim} Δ_launch")
        ax.plot(batches, [p[2] for p in points], marker="^", ls="--",
                label=f"{demo} dim={dim} B")
    if data:
        ax.legend(fontsize=7)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Time (µs)")
    ax.set_title("Batch Transition Sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
