import re
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import commit

from egcm.paths import ROOT, STAGES
from egcm.provenance import code_hash, inputs_sha256
from egcm.report import Report, write_report
from scripts.gen_status import render


def put(root, stage, **kw):
    d = root / "artifacts" / stage
    d.mkdir(parents=True, exist_ok=True)
    write_report(Report(stage=stage, **kw), d)


def test_justfile_stage_list_matches_paths():
    m = re.search(r'^stages := "([^"]*)"', (ROOT / "justfile").read_text(), re.M)
    assert tuple(m.group(1).split()) == STAGES


def test_status_with_no_reports(repo):
    text = render(repo)
    assert "No stages run yet" in text and "just stage 0" in text


def test_status_passed_stale_and_next(repo):
    put(
        repo, "s0_reference", status="passed", passed=True, git_sha="a" * 40,
        code_hash=code_hash("s0_reference", repo), inputs_sha256=inputs_sha256("s0_reference", repo),
    )  # fmt: skip
    text = render(repo)
    row = next(line for line in text.splitlines() if line.startswith("| s0_reference"))
    assert "| passed | yes | no |" in row
    assert "Next: `just stage 1`" in text

    (repo / "egcm" / "paths.py").write_text("X = 2\n")
    commit(repo)
    row = next(line for line in render(repo).splitlines() if line.startswith("| s0_reference"))
    assert "| passed | yes | yes |" in row  # stale
    assert "Next: `just stage 0`" in render(repo)


def test_status_failed_and_invalid(repo):
    put(repo, "s0_reference", status="failed", failures=["min_a: 0.1 not >= 0.5"])
    (repo / "artifacts" / "s1_rewrite").mkdir(parents=True)
    (repo / "artifacts" / "s1_rewrite" / "report.json").write_text("{not json")
    text = render(repo)
    assert "min_a: 0.1 not >= 0.5" in text
    assert "| s1_rewrite | invalid |" in text


@pytest.mark.skipif(shutil.which("just") is None or shutil.which("uv") is None, reason="just and uv needed")
def test_just_stage_errors_are_clear():
    def run(n):
        return subprocess.run(["just", "stage", n], cwd=ROOT, capture_output=True, text=True)

    for bad in ("9", "x", "1; ls"):
        r = run(bad)
        assert r.returncode == 2 and "error:" in r.stderr, (bad, r.stderr)
    missing = next((n for n in "01234567" if not (Path(ROOT) / "stages" / STAGES[int(n)] / "run.py").exists()), None)
    if missing is not None:  # once every stage has a run.py there is nothing left to check
        r = run(missing)
        assert r.returncode == 2 and "does not exist yet" in r.stderr


def test_status_shows_key_metrics_and_committed_results(repo):
    put(repo, "s0_reference", status="failed", metrics={"min_a": 0.123456789, "flag": True, "other": 3},
        gate_values={"min_a": 0.5, "flag": True}, failures=["min_a: 0.123456789 not >= 0.5"])
    row = next(line for line in render(repo).splitlines() if line.startswith("| s0_reference"))
    assert "| min_a=0.123457, flag=true |" in row  # gated metrics only, in gate order

    results = repo / "docs" / "results"
    results.mkdir(parents=True)
    write_report(Report(stage="s6_device", status="passed", passed=True, git_sha="b" * 40,
                        metrics={"max_peak_rss_mb": 512.0, "min_devices": 1},
                        gate_values={"max_peak_rss_mb": 800, "min_devices": 1}), results)
    (results / "report.json").rename(results / "s6_iPhone16,1.json")
    (results / "broken.json").write_text("{")
    text = render(repo)
    assert "## Committed results (docs/results/)" in text
    assert "| s6_iPhone16,1.json | s6_device | passed | yes | bbbbbbb | max_peak_rss_mb=512, min_devices=1 |" in text
    assert "| broken.json | - | invalid |" in text


def test_status_with_only_committed_results(repo):
    results = repo / "docs" / "results"
    results.mkdir(parents=True)
    write_report(Report(stage="s6_device", status="passed", passed=True), results)
    text = render(repo)
    assert "No stages run yet" in text and "| report.json | s6_device | passed |" in text


def test_fmt_never_rounds_up_to_one():
    from scripts.gen_status import fmt

    assert fmt(0.9999999814435638) == "1 - 1.9e-08"
    assert fmt(1.0) == "1" and fmt(0.994037) == "0.994037" and fmt(True) == "true" and fmt(None) == "none"
    assert fmt(float("inf")) == "inf" and fmt(3) == "3"
