import pytest
import os
import csv
from decodebench.bytes_model import StageTrace
from decodebench.report import Report

def test_report_success(tmp_path):
    # Setup mock data for F1
    traces = [
        StageTrace("rmsnorm", reads=[8192, 8192], write=8192, is_final=False),
        StageTrace("gemv", reads=[33554432, 8192], write=8192, is_final=True)
    ]
    # 30 trials
    stream_us = [20.0] * 30
    graph_us = [16.0] * 30
    
    rep = Report(
        name="test_f1",
        stream_us=stream_us,
        graph_us=graph_us,
        traces=traces,
        byte_threshold=0.01
    )
    
    verd = rep.verdict()
    assert verd.bound == "launch-bound"
    assert verd.delta_launch_ci is not None
    
    # Check CSV export
    csv_file = tmp_path / "test_report.csv"
    rep.to_csv(str(csv_file))
    
    assert os.path.exists(csv_file)
    with open(csv_file, "r") as f:
        reader = list(csv.reader(f))
        assert reader[0] == ["name", "variant", "trial", "us_per_invocation"]
        # 30 stream + 30 graph = 60 rows (+ 1 header)
        assert len(reader) == 61
        assert reader[1] == ["test_f1", "stream", "0", "20.0"]
        assert reader[31] == ["test_f1", "graph", "0", "16.0"]

def test_report_with_fused(tmp_path):
    traces = [
        StageTrace("rmsnorm", reads=[8192, 8192], write=8192, is_final=False),
        StageTrace("gemv", reads=[33554432, 8192], write=8192, is_final=True)
    ]
    stream_us = [20.0] * 30
    graph_us = [16.0] * 30
    fused_us = [12.0] * 30
    
    rep = Report(
        name="test_f1",
        stream_us=stream_us,
        graph_us=graph_us,
        traces=traces,
        fused_us=fused_us,
        byte_threshold=0.01
    )
    
    verd = rep.verdict()
    assert verd.t_fused == 12.0
    assert verd.residual_us is not None
    
    csv_file = tmp_path / "test_report.csv"
    rep.to_csv(str(csv_file))
    with open(csv_file, "r") as f:
        reader = list(csv.reader(f))
        # 30 stream + 30 graph + 30 fused = 90 rows (+ 1 header)
        assert len(reader) == 91
        assert reader[61] == ["test_f1", "fused", "0", "12.0"]

def test_report_capture_failure():
    rep = Report(
        name="test_fail",
        stream_us=[20.0],
        graph_us=[],
        traces=[],
        graph_ok=False,
        graph_skip_reason="Host sync detected"
    )
    with pytest.raises(RuntimeError, match="capture failed for 'test_fail': Host sync detected"):
        rep.verdict()
