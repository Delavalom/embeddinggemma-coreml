import math

import numpy as np
import pytest

from egcm.provenance import code_hash, inputs_sha256, is_stale
from egcm.report import Report, read_report, write_report
from egcm.stage import run_stage


def report_path(repo, stage="s0_reference"):
    return repo / "artifacts" / stage / "report.json"


def seed(repo, stage="s0_reference", **kw):
    d = report_path(repo, stage).parent
    d.mkdir(parents=True, exist_ok=True)
    write_report(Report(stage=stage, **kw), d)


def ok(r):
    r.metrics["min_a"] = 0.9
    r.details["note"] = "hi"


def test_pass_writes_stamped_report(repo):
    r = run_stage("s0_reference", ok, root=repo)
    back = read_report(report_path(repo))
    assert back == r
    assert back.status == "passed" and back.passed and back.failures == []
    assert back.gate_values == {"min_a": 0.5}
    assert len(back.git_sha) == 40 and back.git_dirty is False
    assert back.code_hash == code_hash("s0_reference", repo)
    assert back.inputs_sha256 == inputs_sha256("s0_reference", repo)
    assert back.started_at and back.finished_at
    assert not is_stale(back, repo)


def test_running_report_replaces_old_pass_before_fn_runs(repo):
    seed(repo, status="passed", passed=True)
    seen = {}

    def fn(r):
        seen["during"] = read_report(report_path(repo))
        ok(r)

    run_stage("s0_reference", fn, root=repo)
    assert seen["during"].status == "running" and seen["during"].passed is False
    assert seen["during"].started_at


def test_gate_failure_exits_nonzero_and_records_failed(repo):
    def fn(r):
        r.metrics["min_a"] = 0.1

    with pytest.raises(SystemExit) as e:
        run_stage("s0_reference", fn, root=repo)
    assert e.value.code == 1
    back = read_report(report_path(repo))
    assert back.status == "failed" and not back.passed and "min_a" in back.failures[0]


def test_exception_writes_error_with_traceback(repo):
    seed(repo, status="passed", passed=True)

    def fn(r):
        raise RuntimeError("boom")

    with pytest.raises(SystemExit) as e:
        run_stage("s0_reference", fn, root=repo)
    assert e.value.code == 1
    back = read_report(report_path(repo))
    assert back.status == "error" and not back.passed
    assert "boom" in back.failures[0] and "RuntimeError" in back.details["traceback"]


def test_nan_metric_fails_and_is_written(repo):
    def fn(r):
        r.metrics["min_a"] = math.nan

    with pytest.raises(SystemExit):
        run_stage("s0_reference", fn, root=repo)
    assert read_report(report_path(repo)).status == "failed"


def test_numpy_scalars_are_accepted(repo):
    def fn(r):
        r.metrics["min_a"] = np.float32(0.9)

    assert run_stage("s0_reference", fn, root=repo).passed


def test_unserializable_details_still_leave_an_error_report(repo):
    def fn(r):
        ok(r)
        r.details["bad"] = object()

    with pytest.raises(SystemExit):
        run_stage("s0_reference", fn, root=repo)
    assert read_report(report_path(repo)).status == "error"


def test_stage_without_gates_is_an_error(repo):
    (repo / "gates.toml").write_text("[s1_rewrite]\nmin_a = 0.5\n")
    with pytest.raises(SystemExit):
        run_stage("s0_reference", ok, root=repo)
    assert read_report(report_path(repo)).status == "error"


def test_upstream_must_have_passed(repo):
    with pytest.raises(SystemExit):  # s0 has not run
        run_stage("s1_rewrite", ok, root=repo)
    assert read_report(report_path(repo, "s1_rewrite")).status == "error"

    seed(repo, status="failed")
    with pytest.raises(SystemExit):
        run_stage("s1_rewrite", ok, root=repo)

    run_stage("s0_reference", ok, root=repo)
    assert run_stage("s1_rewrite", ok, root=repo).passed
