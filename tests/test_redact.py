"""No absolute local paths in what is committed or published (reports, docs/results/, docs/RESULTS.md, the upload)."""

import json
from pathlib import Path

import pytest

from egcm.paths import ROOT
from egcm.redact import files_with_home_paths, home_paths, relativize, relativize_text
from egcm.report import Report, read_report, write_report
from scripts.save_result import save
from stages.s7_publish import run as s7


def test_relativize_paths_in_and_out_of_the_repo():
    root, home = Path("/Users/me/labs/repo"), Path("/Users/me")
    text = json.dumps({"a": "/Users/me/labs/repo/artifacts/s4/x.aar", "b": "/Users/me/labs/repo",
                       "c": "/Users/me/labs/repo-other/y", "d": "/Users/me/.cache/e5", "e": "/Users/me",
                       "f": "File \"/Users/me/labs/repo/egcm/x.py\", line 3"})
    out = json.loads(relativize_text(text, root, home))
    assert out == {"a": "artifacts/s4/x.aar", "b": ".", "c": "~/labs/repo-other/y", "d": "~/.cache/e5", "e": "~",
                   "f": "File \"egcm/x.py\", line 3"}
    assert home_paths(json.dumps(out)) == [] and relativize({"k": ["/Users/me/z"]}, root, home) == {"k": ["~/z"]}
    assert home_paths("/home/bob/x and /Users/alice") == ["/Users/alice", "/home/bob"]


def test_report_writer_stores_repo_relative_paths(tmp_path):
    r = Report(stage="s6_device", details={"package": str(ROOT / "artifacts" / "s4_compress" / "x.mlpackage"),
                                           "cache": str(Path.home() / "Library" / "Caches")},
               failures=[f"boom in {ROOT / 'egcm' / 'stage.py'}"])
    text = write_report(r, tmp_path).read_text()
    assert home_paths(text) == [] and str(ROOT) not in text
    back = read_report(tmp_path / "report.json")
    assert back.details == {"package": "artifacts/s4_compress/x.mlpackage", "cache": "~/Library/Caches"}
    assert back.failures == ["boom in egcm/stage.py"]


def test_save_result_writes_no_home_path(tmp_path):
    d = tmp_path / "artifacts" / "s6_device"
    d.mkdir(parents=True)
    raw = {"stage": "s6_device", "details": {"devices": {"iPhone16,1": {}}, "package": str(tmp_path / "pkg"),
                                             "log": str(Path.home() / "x.log")}}
    (d / "report.json").write_text(json.dumps(raw))  # an old report, written before the writer relativized
    out = save("s6_device", root=tmp_path)
    assert out == tmp_path / "docs" / "results" / "s6_iPhone16,1.json"
    data = json.loads(out.read_text())
    assert data["details"]["package"] == "pkg" and data["details"]["log"] == "~/x.log"
    assert home_paths(out.read_text()) == []


def test_committed_results_have_no_home_paths():
    """Fails as soon as a report or doc with an absolute home path is committed under docs/results/ or as RESULTS.md."""
    committed = [ROOT / "docs" / "RESULTS.md", *(ROOT / "docs" / "results").glob("*.json")]
    assert files_with_home_paths(committed) == {}


def test_s7_preflight_refuses_home_paths_and_undocumented_repos(tmp_path):
    (tmp_path / "docs" / "results").mkdir(parents=True)
    (tmp_path / "docs" / "RESULTS.md").write_text("| x | /Users/someone/labs/repo/artifacts |\n")
    (tmp_path / "docs" / "results" / "s6_a.json").write_text('{"p": "/home/ci/x"}')
    (tmp_path / "README.md").write_text(f"on the Hub at {s7.DOCUMENTED_HF_REPO}\n")
    pub = s7.Publisher(run=lambda cmd: s7.subprocess.CompletedProcess(cmd, 1, "", ""))
    problems = s7.preflight(tmp_path, "v1.0.0", pub, "someone/else")
    assert any("README.md does not name the Hugging Face repo someone/else" in p for p in problems)
    assert any(p.startswith("docs/RESULTS.md names ['/Users/someone']") for p in problems)
    assert any(p.startswith("docs/results/s6_a.json names ['/home/ci']") for p in problems)
    problems = s7.preflight(tmp_path, "v1.0.0", pub, s7.DOCUMENTED_HF_REPO)
    assert not any("does not name" in p for p in problems)


@pytest.mark.parametrize("doc", s7.REPO_DOCS)
def test_docs_name_the_documented_repo(doc):
    assert s7.DOCUMENTED_HF_REPO in (ROOT / doc).read_text()
