"""Robust statistics for right-skewed kernel timings. No t-tests."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class Summary:
    median: float
    p25: float
    p75: float
    n: int
    @property
    def iqr(self) -> float:
        return self.p75 - self.p25

def summarize(samples: Sequence[float]) -> Summary:
    a = np.asarray(samples, dtype=np.float64)
    if a.size == 0:
        raise ValueError("summarize() requires at least one sample")
    return Summary(median=float(np.median(a)), p25=float(np.percentile(a, 25)),
                   p75=float(np.percentile(a, 75)), n=int(a.size))

def bootstrap_diff_ci(a, b, n_resamples: int = 10000, ci: float = 0.95,
                      seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap CI of median(a) - median(b). Returns (lo, hi, point)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise ValueError("bootstrap_diff_ci() requires non-empty samples")
    idx_a = rng.integers(0, a.size, size=(n_resamples, a.size))
    idx_b = rng.integers(0, b.size, size=(n_resamples, b.size))
    diffs = np.median(a[idx_a], axis=1) - np.median(b[idx_b], axis=1)
    alpha = (1.0 - ci) / 2.0
    return (float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1.0 - alpha)),
            float(np.median(a) - np.median(b)))
