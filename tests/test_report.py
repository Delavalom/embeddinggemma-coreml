import json
import math

import pydantic
import pytest

from egcm.report import Report, read_report, write_report


def test_roundtrip(tmp_path):
    r = Report(stage="s1_rewrite", status="passed", passed=True, metrics={"min_cosine_text": 0.99995, "ok": True})
    p = write_report(r, tmp_path)
    assert p.name == "report.json"
    assert read_report(p) == r
    assert not list(tmp_path.glob("*.tmp"))


def test_extra_fields_forbidden():
    with pytest.raises(pydantic.ValidationError):
        Report(stage="s1_rewrite", pass_=True)


def test_defaults_are_not_passing():
    r = Report(stage="s1_rewrite")
    assert r.status == "running" and r.passed is False and r.started_at is None


def test_nonfinite_metrics_roundtrip_as_strings(tmp_path):
    r = Report(stage="s1_rewrite", metrics={"a": math.nan, "b": math.inf, "c": -math.inf, "d": 1.5})
    p = write_report(r, tmp_path)

    def reject(constant):
        raise AssertionError(f"report.json contains bare {constant}")

    raw = json.loads(p.read_text(), parse_constant=reject)  # strict JSON: no NaN/Infinity tokens
    assert raw["metrics"]["a"] == "NaN" and raw["metrics"]["b"] == "Infinity"
    back = read_report(p)
    assert math.isnan(back.metrics["a"])
    assert back.metrics["b"] == math.inf and back.metrics["c"] == -math.inf
    assert back.metrics["d"] == 1.5


def test_write_replaces_existing_report(tmp_path):
    write_report(Report(stage="s1_rewrite", status="passed", passed=True), tmp_path)
    write_report(Report(stage="s1_rewrite"), tmp_path)
    assert read_report(tmp_path / "report.json").status == "running"
