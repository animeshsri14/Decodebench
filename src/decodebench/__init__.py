"""DecodeBench: Quantify launch overhead vs. byte-elimination in LLM decode fusion."""

__version__ = "0.1.0"

# Lazy torch import — torch is not a hard dependency for CPU-only paths.
# GPU-only modules (sequence, timing, graph, demos) import torch inside methods.

from decodebench.bytes_model import StageTrace, eliminable_bytes, total_bytes
from decodebench.report import Report
from decodebench.stats import Summary, bootstrap_diff_ci, summarize
from decodebench.verdict import Verdict, compute_verdict
from decodebench.sequence import Sequence

__all__ = [
    "__version__",
    "StageTrace",
    "eliminable_bytes",
    "total_bytes",
    "Verdict",
    "compute_verdict",
    "Report",
    "Summary",
    "bootstrap_diff_ci",
    "summarize",
    "Sequence",
]
