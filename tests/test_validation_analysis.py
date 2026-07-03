import csv
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from validation.analysis.compare import (
    Checks,
    f4_delta_detectable,
    relative_delta_matches,
    residual_matches_model,
    residual_term,
    sign_corroborated,
)
from validation.analysis.parse_ncu import find_round_size


def test_residual_gate_is_two_sided():
    assert residual_matches_model(100.0, 99.0, 1.0)[0]
    assert not residual_matches_model(100.0, 80.0, 1.0)[0]
    assert not residual_matches_model(100.0, 120.0, 1.0)[0]


def test_residual_term_sign_convention():
    # Fused faster than graph by more than B -> positive S
    assert residual_term(384.0, 318.0, 11.6) > 0
    # Fused slower than graph -> negative S
    assert residual_term(220.0, 253.0, 0.03) < 0


def test_sign_corroboration_requires_agreement_outside_noise():
    assert sign_corroborated(66.0, 136.0) == "PASS"       # both positive
    assert sign_corroborated(-102.0, -154.0) == "PASS"    # both negative
    assert sign_corroborated(66.0, -136.0) == "FAIL"      # disagreement, both large
    # Either magnitude inside the 5 us band: no direction can be established
    # for that instrument, so no corroboration claim either way (2026-07-03
    # change control: these were previously vacuous PASSes).
    assert sign_corroborated(-14.0, -2.0) == "INDETERMINATE"
    assert sign_corroborated(3.0, -136.0) == "INDETERMINATE"
    assert sign_corroborated(2.0, 1.0) == "INDETERMINATE"  # both in band (tie)


def test_f4_delta_detectability_floor():
    # 1 MB analytic delta on ~90 MB totals: below the 5% floor
    assert not f4_delta_detectable(1.0e6, 97.4e6, 90.4e6)
    # 6 MB analytic delta on ~90 MB totals: detectable
    assert f4_delta_detectable(6.0e6, 97.4e6, 90.4e6)
    # Missing totals: never detectable
    assert not f4_delta_detectable(6.0e6, 0.0, 90.4e6)


def test_delta_gate_is_two_sided():
    assert relative_delta_matches(1.2, 1.0, 0.5)
    assert not relative_delta_matches(2.0, 1.0, 0.5)
    assert not relative_delta_matches(0.4, 1.0, 0.5)


def test_warnings_are_not_a_valid_pass():
    checks = Checks()
    checks.add("dev", "missing", "WARN")
    assert checks.overall() == "INCOMPLETE"


def test_bootstrap_s_ci_deterministic_and_brackets_point_estimate():
    from validation.analysis.compare import bootstrap_s_ci
    graph = [100.0 + 0.1 * i for i in range(30)]
    fused = [90.0 + 0.1 * i for i in range(30)]
    ci1 = bootstrap_s_ci(graph, fused, elim_frac=0.03)
    ci2 = bootstrap_s_ci(graph, fused, elim_frac=0.03)
    assert ci1 == ci2  # fixed seed -> reproducible reports
    lo, hi = ci1
    point = statistics.median(graph) * 0.97 - statistics.median(fused)
    assert lo <= point <= hi
    assert bootstrap_s_ci([], fused, 0.03) is None


def test_indeterminate_neither_passes_nor_gates():
    checks = Checks()
    checks.add("a", "tie-cell", "INDETERMINATE")
    checks.add("b", "real-pass", "PASS")
    c = checks.counts()
    assert c["INDETERMINATE"] == 1 and c["PASS"] == 1
    # Does not gate (no FAIL/WARN -> overall PASS) but is counted separately,
    # never inflating the PASS tally.
    assert checks.overall() == "PASS"


def test_ncu_round_detection_rejects_nonperiodic_sequence():
    assert find_round_size([(0, "a"), (1, "b"), (2, "a"), (3, "b")]) == 2
    with pytest.raises(ValueError, match="not periodic"):
        find_round_size([(0, "a"), (1, "b"), (2, "a"), (3, "c")])


def test_compare_fails_closed_on_empty_inputs(tmp_path):
    timing = tmp_path / "timing.csv"
    ncu = tmp_path / "ncu.csv"
    report = tmp_path / "report.md"
    timing.write_text(
        "gpu_name,fusion,variant,dim,batch,trial,iters,us_per_invocation,correctness_ok,timestamp\n"
    )
    with ncu.open("w", newline="") as fh:
        csv.writer(fh).writerow(
            ["fusion", "variant", "dram_bytes_read", "dram_bytes_write"]
        )
    script = Path(__file__).parents[1] / "validation" / "analysis" / "compare.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--timing-csv", str(timing),
         "--ncu-csv", str(ncu), "--output", str(report)],
        text=True, capture_output=True,
    )
    assert proc.returncode != 0
    assert "Overall: FAIL" in report.read_text()
