import math

import pytest

from decodebench.timing import n_weight_replicas, time_callable


def test_replica_sizing_validation():
    assert n_weight_replicas(1024, l2_bytes=48 * 1024 * 1024) == 8
    with pytest.raises(ValueError, match="weight_bytes"):
        n_weight_replicas(0, l2_bytes=1024)
    with pytest.raises(ValueError, match="l2_bytes"):
        n_weight_replicas(1024, l2_bytes=-1)


def test_timing_arguments_fail_before_torch_is_needed():
    with pytest.raises(ValueError, match="trials"):
        time_callable(lambda: None, trials=0)
    with pytest.raises(ValueError, match="target_ms"):
        time_callable(lambda: None, target_ms=math.nan)
    with pytest.raises(ValueError, match="warmup"):
        time_callable(lambda: None, warmup=-1)
