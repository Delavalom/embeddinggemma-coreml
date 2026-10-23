import math

import pytest

from egcm.gates import evaluate, gates, ops_for, targets
from egcm.paths import STAGES
from egcm.report import Report, read_report, write_report


def toml(tmp_path, text):
    p = tmp_path / "gates.toml"
    p.write_text(text)
    return p


def run(metrics, text, stage="s_x", tmp_path=None):
    p = toml(tmp_path, text)
    return evaluate(Report(stage=stage, metrics=metrics), stage, path=p)


def test_pass_sets_gate_values_from_toml(tmp_path):
    r = run({"min_cosine_text": 0.99995}, "[s_x]\nmin_cosine_text = 0.9999\n", tmp_path=tmp_path)
    assert r.passed and r.status == "passed" and r.failures == []
    assert r.gate_values == {"min_cosine_text": 0.9999}


def test_min_fail_records_reason(tmp_path):
    r = run({"min_ane_fraction": 0.8}, "[s_x]\nmin_ane_fraction = 0.95\n", tmp_path=tmp_path)
    assert not r.passed and r.status == "failed"
    assert "min_ane_fraction" in r.failures[0]


def test_max_gate(tmp_path):
    t = "[s_x]\nmax_download_mb = 300\n"
    assert not run({"max_download_mb": 310.0}, t, tmp_path=tmp_path).passed
    assert run({"max_download_mb": 300}, t, tmp_path=tmp_path).passed


def test_bool_gate(tmp_path):
    t = "[s_x]\ndeterministic = true\n"
    assert run({"deterministic": True}, t, tmp_path=tmp_path).passed
    assert not run({"deterministic": False}, t, tmp_path=tmp_path).passed
    assert not run({"deterministic": 1}, t, tmp_path=tmp_path).passed  # not a bool


def test_bad_metrics_are_failures_not_exceptions(tmp_path):
    t = "[s_x]\nmin_a = 0.5\n"
    for bad in ({}, {"min_a": None}, {"min_a": "high"}, {"min_a": True}, {"min_a": math.nan}, {"min_a": math.inf}):
        r = run(bad, t, tmp_path=tmp_path)
        assert not r.passed and len(r.failures) == 1, bad


def test_nan_metric_roundtrips_and_fails(tmp_path):
    r = run({"min_a": math.nan}, "[s_x]\nmin_a = 0.5\n", tmp_path=tmp_path)
    p = write_report(r, tmp_path)
    back = read_report(p)
    assert math.isnan(back.metrics["min_a"])
    assert not back.passed
    assert not evaluate(back, "s_x", path=tmp_path / "gates.toml").passed


def test_missing_or_empty_stage_raises(tmp_path):
    p = toml(tmp_path, "[s_x]\nmin_a = 1\n[s_empty]\n")
    with pytest.raises(ValueError):
        evaluate(Report(stage="s_none"), "s_none", path=p)
    with pytest.raises(ValueError):
        evaluate(Report(stage="s_empty"), "s_empty", path=p)


def test_strict_operator_inference(tmp_path):
    for bad in ("cosine = 0.9", "threshold = 1", 'min_a = "x"'):
        with pytest.raises(ValueError):
            gates("s_x", toml(tmp_path, f"[s_x]\n{bad}\n"))
    assert ops_for({"min_a": 1, "max_b": 2, "flag": True}) == {"min_a": ">=", "max_b": "<=", "flag": "=="}


def test_targets_are_not_gated(tmp_path):
    p = toml(tmp_path, "[s_x]\nmin_a = 1\n[s_x.targets]\ntarget_ms = 60\n")
    assert gates("s_x", p) == {"min_a": 1}
    assert targets("s_x", p) == {"target_ms": 60}
    assert targets("s_x", toml(tmp_path, "[s_x]\nmin_a = 1\n")) == {}


def test_repo_gates_toml_is_valid():
    for stage in STAGES:
        g = gates(stage)
        assert g, stage
        ops_for(g)
    assert targets("s6_device") == {"target_ms_per_image_70": 60}
    assert "target_ms_per_image_70" not in gates("s6_device")
