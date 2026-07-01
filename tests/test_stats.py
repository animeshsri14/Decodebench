import pytest
import numpy as np
from decodebench.stats import summarize, bootstrap_diff_ci

def test_summarize():
    summary = summarize([10, 12, 11, 13, 100])
    assert summary.median == 12.0
    assert summary.p25 == 11.0
    assert summary.p75 == 13.0
    assert summary.iqr == 2.0
    assert summary.n == 5

    with pytest.raises(ValueError, match="requires at least one sample"):
        summarize([])

def test_bootstrap_diff_ci():
    rng = np.random.default_rng(42)
    a = rng.normal(100, 2, 200)
    b = rng.normal(90, 2, 200)
    lo, hi, point = bootstrap_diff_ci(a, b, n_resamples=1000, seed=42)
    assert lo > 0
    assert lo < point < hi

    # Identical distributions -> CI straddles 0
    a2 = rng.normal(100, 2, 200)
    lo2, hi2, point2 = bootstrap_diff_ci(a, a2, n_resamples=1000, seed=42)
    assert lo2 < 0 < hi2

    with pytest.raises(ValueError, match="requires non-empty samples"):
        bootstrap_diff_ci([], [1.0])
