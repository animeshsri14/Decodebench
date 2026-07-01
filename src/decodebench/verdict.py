"""Verdict: launch overhead (measured) vs eliminable-byte ceiling (analytic)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

DEFAULT_BYTE_THRESHOLD = 0.01   # RI-1; calibrated in design §5.2

@dataclass(frozen=True)
class Verdict:
    t_stream: float          # us, median
    t_graph: float           # us, median (the floor once CUDA Graphs are on)
    delta_launch: float      # us = t_stream - t_graph ("M", measured)
    b_ceiling: float         # us, upper bound on byte-elimination gain beyond graphs ("B")
    achieved_bw: float       # bytes/us
    total_bytes: int
    eliminable_bytes: int
    byte_threshold: float
    bound: str               # "launch-bound" | "byte-bound"
    delta_launch_ci: Optional[tuple[float, float]] = None
    t_fused: Optional[float] = None
    residual_us: Optional[float] = None

    def render(self) -> str:
        ci = ""
        if self.delta_launch_ci is not None:
            lo, hi = self.delta_launch_ci
            ci = f" (95% CI [{lo:.2f}, {hi:.2f}])"
        advice = ("enable CUDA Graphs; hand-fusion is not worth the maintenance cost."
                  if self.bound == "launch-bound" else
                  "hand-fusion may be worth it (savings come from eliminated bytes, not launches).")
        lines = [
            f"CUDA Graphs eliminate {self.delta_launch:.2f} us here{ci} (measured).",
            f"Fusion can save at most {self.b_ceiling:.2f} us from eliminable intermediate bytes "
            "(analytic byte ceiling).",
            f"Floor with graphs on (t_graph): {self.t_graph:.2f} us.",
        ]
        if self.t_fused is not None and self.residual_us is not None:
            lines.append(f"Measured fused latency: {self.t_fused:.2f} us.")
            lines.append(f"Decomposition: launch overhead {self.delta_launch:.2f} us + "
                         f"eliminable bytes {self.b_ceiling:.2f} us + "
                         f"efficiency residual {self.residual_us:.2f} us = "
                         f"{self.delta_launch + self.b_ceiling + self.residual_us:.2f} us "
                         f"(t_stream - t_fused).")
        lines.append(f"Verdict: {self.bound.upper()} -> {advice}")
        return "\n".join(lines)

def compute_verdict(t_stream: float, t_graph: float, total_bytes: int,
                    eliminable_bytes: int, byte_threshold: float = DEFAULT_BYTE_THRESHOLD,
                    delta_launch_ci: Optional[tuple[float, float]] = None,
                    t_fused: Optional[float] = None) -> Verdict:
    if t_graph <= 0:
        raise ValueError("t_graph must be positive (us)")
    if t_fused is not None and t_fused <= 0:
        raise ValueError("t_fused must be positive (us) when provided")
    achieved_bw = total_bytes / t_graph
    b_ceiling = eliminable_bytes / achieved_bw
    bound = "byte-bound" if b_ceiling >= byte_threshold * t_graph else "launch-bound"
    residual_us = None if t_fused is None else (t_graph - t_fused) - b_ceiling
    return Verdict(t_stream, t_graph, t_stream - t_graph, b_ceiling, achieved_bw,
                   total_bytes, eliminable_bytes, byte_threshold, bound, delta_launch_ci,
                   t_fused, residual_us)
