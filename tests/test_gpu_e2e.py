import pytest
import warnings

# torch is only needed for the GPU e2e path and is not a core dependency
# (pyproject declares only numpy + matplotlib). Skip the whole module cleanly
# on stock installs instead of failing collection with ModuleNotFoundError.
torch = pytest.importorskip("torch")
from decodebench.sequence import Sequence
from decodebench.graph import try_capture, Captured
from decodebench.timing import n_weight_replicas, time_callable
from decodebench.demos.llama_decode import build_demo

@pytest.mark.gpu
def test_n_weight_replicas():
    assert n_weight_replicas(1024, l2_bytes=48*1024*1024) >= 4

@pytest.mark.gpu
def test_sequence_trace_bytes():
    # Construct a simple sequence and trace it
    d = 256
    x = torch.randn(1, d, dtype=torch.float16, device="cuda")
    g = torch.randn(d, dtype=torch.float16, device="cuda")
    
    seq = Sequence("test_trace")
    @seq.stage
    def rmsnorm(x, g):
        return x * g
        
    traces = seq.trace({"x": x, "g": g})
    assert len(traces) == 1
    assert traces[0].name == "rmsnorm"
    assert traces[0].reads == [x.nbytes, g.nbytes]
    assert traces[0].write == x.nbytes

@pytest.mark.gpu
def test_sequence_trace_fanout_and_unsafe_tensor_forms():
    x = torch.randn(16, device="cuda")
    seq = Sequence("fanout")

    @seq.stage
    def source(x):
        return x.clone()

    @seq.stage
    def left(source):
        return source + 1

    @seq.stage
    def right(source):
        return source + 2

    @seq.stage
    def merge(left, right):
        return left + right

    traces = seq.trace({"x": x})
    assert traces[0].consumers == 2

    alias = Sequence("alias")
    @alias.stage
    def inplace(x):
        return x.add_(1)
    with pytest.raises(ValueError, match="aliases"):
        alias.trace({"x": x.clone()})

    strided = Sequence("strided")
    @strided.stage
    def copy(x):
        return x.clone()
    with pytest.raises(ValueError, match="non-contiguous"):
        strided.trace({"x": torch.ones(4, 4, device="cuda").t()})

    hidden_weight = torch.ones(16, device="cuda")
    closure = Sequence("closure")
    @closure.stage
    def hidden(x):
        return x * hidden_weight
    with pytest.raises(ValueError, match="captures tensor"):
        closure.trace({"x": x})

@pytest.mark.gpu
def test_replica_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="does not match replica 0"):
        Sequence._validate_replicas([
            {"x": torch.ones(2, device="cuda")},
            {"x": torch.ones(3, device="cuda")},
        ])

@pytest.mark.gpu
def test_cuda_graph_capture_and_replay():
    x = torch.randn(1, 256, dtype=torch.float16, device="cuda")
    y = torch.zeros(1, 256, dtype=torch.float16, device="cuda")
    
    def body():
        y.copy_(x * 2.0)
        
    captured = try_capture(body)
    assert captured.ok is True
    assert captured._graph is not None
    
    # Modify input and replay
    x.fill_(1.0)
    captured.replay()
    assert torch.allclose(y, torch.tensor(2.0, dtype=torch.float16, device="cuda"))

@pytest.mark.gpu
def test_cuda_graph_capture_failure():
    # Body containing host sync (.item()) should fail capture
    x = torch.tensor([1.0], device="cuda")
    
    def body():
        _ = x.item()
        
    captured = try_capture(body)
    assert captured.ok is False
    assert len(captured.reason) > 0
    
    with pytest.raises(RuntimeError, match="cannot replay: capture failed"):
        captured.replay()

@pytest.mark.gpu
def test_stage_temporary_allocation_warning():
    seq = Sequence("test_temp_warn")
    
    @seq.stage
    def alloc_temp(x):
        # Allocate a large internal tensor (>100 KB) that isn't the output
        _temp = torch.randn(100000, dtype=torch.float32, device="cuda")
        return x.clone()
        
    x = torch.randn(10, dtype=torch.float16, device="cuda")
    
    with pytest.warns(UserWarning, match="allocated .* MB internally; byte model undercounts"):
        seq.trace({"x": x})

@pytest.mark.gpu
def test_time_callable_sanity():
    x = torch.tensor([0], dtype=torch.int32, device="cuda")
    def increment():
        x.add_(1)
    
    us_times = time_callable(increment, trials=10, target_ms=5, warmup=5)
    assert len(us_times) == 10
    assert all(t > 0 for t in us_times)

@pytest.mark.gpu
@pytest.mark.parametrize("demo_name,expected_bound", [
    ("f1", "launch-bound"),
    ("f2", "launch-bound"),
    ("f4", "byte-bound")
])
def test_demo_verdicts(demo_name, expected_bound):
    seq, inputs, replicas = build_demo(demo_name, dim=4096, batch=1)
    report = seq.profile(inputs, trials=10, warmup=10, input_replicas=replicas)
    assert report.graph_ok is True
    v = report.verdict()
    assert v.bound == expected_bound
