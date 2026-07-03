"""Verdict: launch overhead (measured) vs eliminable-byte estimate (analytic).

Terminology note (revised): B is a *proportional byte-time estimate*, not a
mathematical upper bound. B = eliminable_bytes / (total_bytes / t_graph)
assumes runtime scales linearly with declared bytes and that eliminated
intermediates see the same effective bandwidth as the dominant traffic. That
assumption fails for launch-, latency-, compute-, or occupancy-bound kernels,
cache-resident traffic, and serialized stage boundaries — all of which are
common in decode. Treat B as "the time the eliminated bytes would take at the
workload's average byte throughput", nothing stronger.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional

DEFAULT_BYTE_THRESHOLD = 0.01   # RI-1; calibrated in design §5.2

@dataclass(frozen=True)
class Verdict:
    t_stream: float          # us, median
    t_graph: float           # us, median (the floor once CUDA Graphs are on)
    delta_launch: float      # us = t_stream - t_graph ("M", measured)
    b_bytes_est: float       # us, proportional byte-time estimate for eliminable bytes ("B")
    achieved_bw: float       # bytes/us
    total_bytes: int
    eliminable_bytes: int
    byte_threshold: float
    bound: str               # "low-byte-opportunity" | "material-byte-opportunity" (analytic
                             # byte-fraction materiality; pre-2026-07-03 versions said
                             # "launch-bound"/"byte-bound", which overstated the diagnosis)
    dominant: str            # "launch" | "bytes" — which measured/estimated term is larger
    delta_launch_ci: Optional[tuple[float, float]] = None
    t_fused: Optional[float] = None
    residual_us: Optional[float] = None

    @property
    def b_ceiling(self) -> float:
        """Deprecated alias for b_bytes_est (the old name overstated the claim)."""
        return self.b_bytes_est

    def render(self) -> str:
        ci = ""
        ci_spans_zero = False
        if self.delta_launch_ci is not None:
            lo, hi = self.delta_launch_ci
            ci = f" (95% CI [{lo:.2f}, {hi:.2f}])"
            ci_spans_zero = lo <= 0.0 <= hi
        advice = ("enable CUDA Graphs first; the declared byte fraction is below the configured fusion threshold."
                  if self.bound == "low-byte-opportunity" else
                  "the declared byte fraction clears the threshold; benchmark a representative fused kernel before committing.")
        launch_line = (
            f"CUDA Graphs eliminate {self.delta_launch:.2f} us here{ci} (measured)."
            if not ci_spans_zero else
            f"Measured launch saving {self.delta_launch:.2f} us{ci} is statistically "
            "indistinguishable from zero (CI spans 0)."
        )
        lines = [
            launch_line,
            f"Eliminable intermediate bytes correspond to ~{self.b_bytes_est:.2f} us at this "
            "workload's average byte throughput (analytic proportional estimate, NOT a strict bound).",
            f"Floor with graphs on (t_graph): {self.t_graph:.2f} us.",
            f"Dominant term: {'launch overhead (measured)' if self.dominant == 'launch' else 'eliminable bytes (estimated)'} "
            f"(delta_launch {self.delta_launch:.2f} us vs B {self.b_bytes_est:.2f} us).",
        ]
        if self.t_fused is not None and self.residual_us is not None:
            lines.append(f"Measured fused latency: {self.t_fused:.2f} us.")
            lines.append(f"Decomposition: launch overhead {self.delta_launch:.2f} us + "
                         f"eliminable bytes {self.b_bytes_est:.2f} us + "
                         f"unexplained residual {self.residual_us:.2f} us = "
                         f"{self.delta_launch + self.b_bytes_est + self.residual_us:.2f} us "
                         f"(t_stream - t_fused).")
        lines.append(f"Verdict: {self.bound.upper()} -> {advice}")
        if self.bound == "material-byte-opportunity" and self.dominant == "launch":
            lines.append(
                "Caution: measured launch savings exceed the byte estimate; the "
                "material-byte-opportunity classification reflects the analytic byte "
                "fraction, not the empirically dominant source of gain.")
        return "\n".join(lines)

def compute_verdict(t_stream: float, t_graph: float, total_bytes: int,
                    eliminable_bytes: int, byte_threshold: float = DEFAULT_BYTE_THRESHOLD,
                    delta_launch_ci: Optional[tuple[float, float]] = None,
                    t_fused: Optional[float] = None) -> Verdict:
    if not math.isfinite(t_stream) or not math.isfinite(t_graph) or t_stream <= 0:
        raise ValueError("t_stream and t_graph must be finite, and t_stream must be positive")
    if t_graph <= 0:
        raise ValueError("t_graph must be positive (us)")
    if t_fused is not None and (not math.isfinite(t_fused) or t_fused <= 0):
        raise ValueError("t_fused must be finite and positive (us) when provided")
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    if not 0 <= eliminable_bytes <= total_bytes:
        raise ValueError("eliminable_bytes must be in [0, total_bytes]")
    if not math.isfinite(byte_threshold) or not 0 <= byte_threshold <= 1:
        raise ValueError("byte_threshold must be finite and in [0, 1]")
    if delta_launch_ci is not None:
        lo, hi = delta_launch_ci
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError("delta_launch_ci must contain finite ordered bounds")
    achieved_bw = total_bytes / t_graph
    b_bytes_est = eliminable_bytes / achieved_bw
    delta_launch = t_stream - t_graph
    # `bound` classifies the analytic eliminable-byte fraction (hardware-independent
    # materiality gate); `dominant` compares it against the *measured* launch term.
    # The labels deliberately claim only byte *opportunity* — the threshold cannot
    # diagnose a chain as compute-, latency-, or occupancy-bound.
    bound = ("material-byte-opportunity" if b_bytes_est >= byte_threshold * t_graph
             else "low-byte-opportunity")
    dominant = "launch" if delta_launch >= b_bytes_est else "bytes"
    residual_us = None if t_fused is None else (t_graph - t_fused) - b_bytes_est
    return Verdict(t_stream, t_graph, delta_launch, b_bytes_est, achieved_bw,
                   total_bytes, eliminable_bytes, byte_threshold, bound, dominant,
                   delta_launch_ci, t_fused, residual_us)
