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
