import pytest
from decodebench.verdict import compute_verdict, Verdict

def test_verdict_launch_bound():
    # launch-bound case (t_stream=20, t_graph=16, F1 bytes: total 33595392, elim 16384)
    v = compute_verdict(
        t_stream=20.0,
        t_graph=16.0,
        total_bytes=33595392,
        eliminable_bytes=16384,
        byte_threshold=0.01
    )
    assert v.bound == "launch-bound"
    assert v.delta_launch == 4.0
    assert abs(v.b_ceiling - (16384 / (33595392 / 16.0))) < 1e-6
    
    # Check render text
    r = v.render()
    assert "CUDA Graphs eliminate 4.00 us here" in r
    assert "from eliminable intermediate bytes" in r
    assert "Verdict: LAUNCH-BOUND" in r

def test_verdict_threshold_respected():
    # Use a 3% ratio (30,000 / 1,000,000)
    # With threshold 0.01 (1%), should be byte-bound
    v_byte = compute_verdict(t_stream=20, t_graph=16, total_bytes=1000000, eliminable_bytes=30000, byte_threshold=0.01)
    assert v_byte.bound == "byte-bound"

    # With threshold 0.05 (5%), should be launch-bound
    v_launch = compute_verdict(t_stream=20, t_graph=16, total_bytes=1000000, eliminable_bytes=30000, byte_threshold=0.05)
    assert v_launch.bound == "launch-bound"

def test_verdict_f4_ratio_regression():
    # RI-1: F4 ratio case at dim=4096, B=1: total_bytes=17317888, eliminable_bytes=524288
    v = compute_verdict(
        t_stream=20.0,
        t_graph=16.0,
        total_bytes=17317888,
        eliminable_bytes=524288,
        byte_threshold=0.01
    )
    # Ratio = 524288 / 17317888 = 3.027% which is > 1% threshold
    assert v.bound == "byte-bound"

def test_verdict_invalid_inputs():
    with pytest.raises(ValueError, match="t_graph must be positive"):
        compute_verdict(t_stream=10.0, t_graph=0.0, total_bytes=100, eliminable_bytes=10)
    with pytest.raises(ValueError, match="t_fused must be positive"):
        compute_verdict(t_stream=10.0, t_graph=8.0, total_bytes=100, eliminable_bytes=10, t_fused=-1.0)

def test_verdict_with_fused():
    v = compute_verdict(
        t_stream=20.0,
        t_graph=16.0,
        total_bytes=33595392,
        eliminable_bytes=16384,
        byte_threshold=0.01,
        t_fused=12.0
    )
    # B = 16384 / (33595392 / 16) = 0.0078 us
    # residual_us = (16.0 - 12.0) - 0.0078 = 3.9922 us
    assert abs(v.residual_us - 3.9922) < 1e-3
    r = v.render()
    assert "Measured fused latency: 12.00 us" in r
    assert "Decomposition:" in r
    assert "efficiency residual 3.99 us" in r
