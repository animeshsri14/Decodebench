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
    if a.ndim != 1:
        raise ValueError("summarize() samples must be one-dimensional")
    if not np.all(np.isfinite(a)):
        raise ValueError("summarize() samples must all be finite")
    return Summary(median=float(np.median(a)), p25=float(np.percentile(a, 25)),
                   p75=float(np.percentile(a, 75)), n=int(a.size))

def bootstrap_diff_ci(a, b, n_resamples: int = 10000, ci: float = 0.95,
                      seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap CI of median(a) - median(b). Returns (lo, hi, point)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise ValueError("bootstrap_diff_ci() requires non-empty samples")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("bootstrap_diff_ci() samples must be one-dimensional")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("bootstrap_diff_ci() samples must all be finite")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be between 0 and 1")
    # Chunk resampling so a researcher passing a long trial vector cannot
    # accidentally allocate O(n_resamples * n_samples) indices at once.
    diffs = np.empty(n_resamples, dtype=np.float64)
    chunk = max(1, 1_000_000 // max(a.size, b.size))
    for start in range(0, n_resamples, chunk):
        stop = min(n_resamples, start + chunk)
        count = stop - start
        idx_a = rng.integers(0, a.size, size=(count, a.size))
        idx_b = rng.integers(0, b.size, size=(count, b.size))
        diffs[start:stop] = (
            np.median(a[idx_a], axis=1) - np.median(b[idx_b], axis=1)
        )
    alpha = (1.0 - ci) / 2.0
    return (float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1.0 - alpha)),
            float(np.median(a) - np.median(b)))
