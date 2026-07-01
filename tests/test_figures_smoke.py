import os
from decodebench.verdict import compute_verdict
from decodebench.figures import (
    plot_verdict_bar,
    plot_predicted_vs_measured,
    plot_cross_arch,
    plot_batch_sweep
)

def test_plot_verdict_bar(tmp_path):
    # Create mock verdicts
    v1 = compute_verdict(t_stream=20.0, t_graph=16.0, total_bytes=33595392, eliminable_bytes=16384)
    v2 = compute_verdict(t_stream=25.0, t_graph=18.0, total_bytes=180437504, eliminable_bytes=44032)
    v4 = compute_verdict(t_stream=30.0, t_graph=16.0, total_bytes=17317888, eliminable_bytes=524288)
    
    verdicts = {"f1": v1, "f2": v2, "f4": v4}
    
    out_png = tmp_path / "verdict_bar.png"
    plot_verdict_bar(verdicts, str(out_png))
    
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 0

def test_plot_predicted_vs_measured(tmp_path):
    out_png = tmp_path / "pred_vs_meas.png"
    plot_predicted_vs_measured({}, str(out_png))
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 0

def test_plot_cross_arch(tmp_path):
    out_png = tmp_path / "cross_arch.png"
    plot_cross_arch({}, str(out_png))
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 0

def test_plot_batch_sweep(tmp_path):
    out_png = tmp_path / "batch_sweep.png"
    plot_batch_sweep([], str(out_png))
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 0
