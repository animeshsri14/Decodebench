import pytest
from model.cost_model import compute_fusion_costs

def test_cost_model_4096():
    costs = compute_fusion_costs(d=4096, b=1, threshold=0.01)
    
    # Verify F1
    f1 = next(c for c in costs if c.name == "F1")
    assert f1.total == 33595392
    assert f1.eliminable == 16384
    assert round(f1.ratio, 6) == 0.000488
    assert f1.predicted == "low-byte-opportunity"

    # Verify F2
    f2 = next(c for c in costs if c.name == "F2")
    assert f2.total == 180481536
    assert f2.eliminable == 88064
    assert round(f2.ratio, 6) == 0.000488
    assert f2.predicted == "low-byte-opportunity"

    # Verify F4
    f4 = next(c for c in costs if c.name == "F4")
    assert f4.total == 17317888
    assert f4.eliminable == 524288
    assert round(f4.ratio, 6) == 0.030274
    assert f4.predicted == "material-byte-opportunity"

def test_cost_model_2048():
    costs = compute_fusion_costs(d=2048, b=1, threshold=0.01)

    # Verify F1
    f1 = next(c for c in costs if c.name == "F1")
    assert f1.total == 8409088
    assert f1.eliminable == 8192
    assert round(f1.ratio, 6) == 0.000974
    assert f1.predicted == "low-byte-opportunity"

    # Verify F2
    f2 = next(c for c in costs if c.name == "F2")
    assert f2.total == 67198976
    assert f2.eliminable == 65536
    assert round(f2.ratio, 6) == 0.000975
    assert f2.predicted == "low-byte-opportunity"

    # Verify F4
    f4 = next(c for c in costs if c.name == "F4")
    assert f4.total == 8658944
    assert f4.eliminable == 262144
    assert round(f4.ratio, 6) == 0.030274
    assert f4.predicted == "material-byte-opportunity"
