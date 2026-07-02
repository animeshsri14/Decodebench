from decodebench.bytes_model import StageTrace, total_bytes, eliminable_bytes

def test_bytes_model_golden_f1():
    # Golden F1 at d=4096 FP16
    traces = [
        StageTrace("rmsnorm", reads=[8192, 8192], write=8192, is_final=False),
        StageTrace("gemv", reads=[33554432, 8192], write=8192, is_final=True)
    ]
    # reads for rmsnorm: 8192 + 8192, write: 8192 -> 24576
    # reads for gemv: 33554432 + 8192, write: 8192 -> 33570816
    # total: 24576 + 33570816 = 33595392
    assert total_bytes(traces) == 33595392
    assert eliminable_bytes(traces) == 16384

def test_single_final_stage():
    traces = [
        StageTrace("gemv", reads=[33554432, 8192], write=8192, is_final=True)
    ]
    assert total_bytes(traces) == 33570816
    assert eliminable_bytes(traces) == 0

def test_fanout_counts_one_read_per_consumer():
    traces = [
        StageTrace("producer", reads=[64], write=128, consumers=2),
        StageTrace("final", reads=[128, 128], write=32, is_final=True, consumers=0),
    ]
    assert total_bytes(traces) == 480
    assert eliminable_bytes(traces) == 384  # write + two downstream reads

def test_invalid_trace_metadata():
    import pytest

    with pytest.raises(ValueError, match="non-negative"):
        StageTrace("bad", reads=[-1], write=1)
    with pytest.raises(ValueError, match="consumers"):
        StageTrace("bad", reads=[], write=1, consumers=-1)
    with pytest.raises(ValueError, match="at least one consumer"):
        StageTrace("dead", reads=[], write=1, consumers=0)
