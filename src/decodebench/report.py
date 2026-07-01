"""Report: bundles raw trials + byte traces; renders the verdict; persists CSV."""
from __future__ import annotations
import csv
from dataclasses import dataclass
from decodebench.bytes_model import StageTrace, eliminable_bytes, total_bytes
from decodebench.stats import bootstrap_diff_ci, summarize
from decodebench.verdict import DEFAULT_BYTE_THRESHOLD, Verdict, compute_verdict

@dataclass
class Report:
    name: str
    stream_us: list[float]
    graph_us: list[float]
    traces: list[StageTrace]
    fused_us: list[float] | None = None
    byte_threshold: float = DEFAULT_BYTE_THRESHOLD
    graph_ok: bool = True
    graph_skip_reason: str = ""

    def verdict(self) -> Verdict:
        if not self.graph_ok:
            raise RuntimeError(
                f"CUDA graph capture failed for '{self.name}': {self.graph_skip_reason}. "
                "Delta_launch is unknown for this chain; no verdict can be emitted.")
        s_stream, s_graph = summarize(self.stream_us), summarize(self.graph_us)
        lo, hi, _ = bootstrap_diff_ci(self.stream_us, self.graph_us)
        t_fused = summarize(self.fused_us).median if self.fused_us is not None else None
        return compute_verdict(t_stream=s_stream.median, t_graph=s_graph.median,
                               total_bytes=total_bytes(self.traces),
                               eliminable_bytes=eliminable_bytes(self.traces),
                               byte_threshold=self.byte_threshold,
                               delta_launch_ci=(lo, hi),
                               t_fused=t_fused)

    def render(self) -> str:
        return self.verdict().render()

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "variant", "trial", "us_per_invocation"])
            for i, t in enumerate(self.stream_us):
                w.writerow([self.name, "stream", i, t])
            for i, t in enumerate(self.graph_us):
                w.writerow([self.name, "graph", i, t])
            if self.fused_us is not None:
                for i, t in enumerate(self.fused_us):
                    w.writerow([self.name, "fused", i, t])
