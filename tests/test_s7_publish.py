"""s7's guards, with fakes for every side effect: no network, no git writes, no Hugging Face calls."""

import subprocess

import pytest
from conftest import commit

from egcm.paths import STAGES
from egcm.provenance import code_hash, inputs_sha256
from egcm.report import Report, read_report, write_report
from stages.s7_publish import run as s7

SHA = "0123456789abcdef0123456789abcdef01234567"


class Untouchable:
    """Any use of the Hugging Face API or a subprocess fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"s7 touched {name} while it should have refused")


def untouchable_publisher():
    def run(cmd):
        raise AssertionError(f"s7 ran {cmd} while it should have refused")

    return s7.Publisher(run=run, hf=Untouchable())


def seed_passing(root, upto=7):
    """Passing, current reports for s0..s(upto-1), each hashed against the tmp repo like run_stage would."""
    for stage in STAGES[:upto]:
        d = root / "artifacts" / stage
        d.mkdir(parents=True, exist_ok=True)
        write_report(Report(stage=stage, status="passed", passed=True, code_hash=code_hash(stage, root),
                            inputs_sha256=inputs_sha256(stage, root)), d)


def test_refuses_without_confirm(repo, capsys):
    seed_passing(repo)
    assert s7.main([], root=repo, env={"HF_REPO_TARGET": "me/model"}, pub=untouchable_publisher()) == 2
    assert "--confirm not given" in capsys.readouterr().out
    assert not (repo / "artifacts" / "s7_publish").exists()  # no report written, an earlier one is never clobbered


@pytest.mark.parametrize("target", [None, "", "no-slash", "a/b/c", "a b/c"])
def test_refuses_without_a_valid_hf_target(repo, target, capsys):
    env = {} if target is None else {"HF_REPO_TARGET": target}
    assert s7.main(["--confirm"], root=repo, env=env, pub=untouchable_publisher()) == 2
    assert "HF_REPO_TARGET" in capsys.readouterr().out
    assert not (repo / "artifacts" / "s7_publish").exists()


def test_confirm_with_missing_upstream_records_an_error_and_publishes_nothing(repo):
    with pytest.raises(SystemExit) as e:
        s7.main(["--confirm"], root=repo, env={"HF_REPO_TARGET": "me/model"}, pub=untouchable_publisher())
    assert e.value.code == 1
    r = read_report(repo / "artifacts" / "s7_publish" / "report.json")
    assert r.status == "error" and not r.passed and "s0_reference has no report" in r.failures[0]
    assert "confirmed" not in r.metrics


def test_upstream_problems_passed_failed_and_stale(repo):
    assert s7.upstream_problems(repo) == [f"{s}: no report" for s in STAGES[:7]]
    seed_passing(repo)
    assert s7.upstream_problems(repo) == []

    p = repo / "artifacts" / "s3_coreml_ane" / "report.json"
    r = read_report(p)
    r.status, r.passed = "failed", False
    write_report(r, p.parent)
    assert s7.upstream_problems(repo) == ["s3_coreml_ane: status 'failed'"] + [
        f"{s}: s3_coreml_ane has status 'failed'; rerun `just stage 3`" for s in STAGES[4:7]]

    seed_passing(repo)
    (repo / "fixtures" / "MANIFEST.sha256").write_text("changed  img.jpg\n")
    commit(repo)
    problems = s7.upstream_problems(repo)
    assert problems[0] == "s0_reference: s0_reference is stale (inputs changed); rerun `just stage 0`"
    assert len(problems) == 7  # everything downstream of s0 is stale too


def test_fill_refuses_on_stale_upstream_even_if_called_directly(repo):
    seed_passing(repo, upto=6)  # no s6 report
    fill = s7.make_fill(repo, "me/model", "v1.0.0", untouchable_publisher())
    r = Report(stage="s7_publish")
    with pytest.raises(RuntimeError, match="s6_device: no report"):
        fill(r)
    assert "confirmed" not in r.metrics


def fake_runner(dirty=False, tag_exists=False):
    calls = []

    def run(cmd):
        calls.append(cmd)
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(cmd, 0, " M x\n" if dirty else "", "")
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0 if tag_exists else 1, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return run, calls


def test_preflight(repo):
    run, _ = fake_runner(dirty=True, tag_exists=True)
    problems = s7.preflight(repo, "v1.0.0", s7.Publisher(run=run, hf=Untouchable()))
    assert len(problems) == 4
    assert "dirty" in problems[0] and "v1.0.0 already exists" in problems[1]
    assert "RESULTS.md" in problems[2] and "s4_compress" in problems[3]


def test_fill_refuses_on_preflight_before_any_network(repo):
    seed_passing(repo)
    run, calls = fake_runner(dirty=True)
    fill = s7.make_fill(repo, "me/model", "v1.0.0", s7.Publisher(run=run, hf=Untouchable()))
    with pytest.raises(RuntimeError, match="dirty"):
        fill(Report(stage="s7_publish"))
    assert all(c[0] == "git" and c[1] in ("status", "rev-parse") for c in calls)


class FakeHF:
    def __init__(self):
        self.calls = []

    def create_repo(self, **kw):
        self.calls.append(("create_repo", kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))
        return type("CommitInfo", (), {"oid": SHA})()


def test_publish_order_with_fakes(tmp_path, monkeypatch):
    (tmp_path / "NOTICE").write_text("notice\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "RESULTS.md").write_text("| parity |\n")
    (tmp_path / s7.DEFAULT_MODEL_SWIFT).parent.mkdir(parents=True)
    monkeypatch.setattr(s7, "build_upload", lambda root, repo, out: "f" * 64)
    hf = FakeHF()
    run, calls = fake_runner()
    details = s7.publish(tmp_path, "me/model", "v1.0.0", s7.Publisher(run=run, hf=hf))
    assert details == {"hf_repo": "me/model", "revision": SHA, "files_json_sha256": "f" * 64, "tag": "v1.0.0"}
    assert hf.calls[0] == ("create_repo", {"repo_id": "me/model", "repo_type": "model", "private": False,
                                           "exist_ok": True})
    assert hf.calls[1][0] == "upload_folder"
    swift = (tmp_path / s7.DEFAULT_MODEL_SWIFT).read_text()
    assert 'defaultRepo = "me/model"' in swift and f'defaultRevision = "{SHA}"' in swift
    assert f'defaultFilesManifestSHA256: String? = "{"f" * 64}"' in swift
    assert [c[:2] for c in calls] == [["git", "add"], ["git", "commit"], ["git", "tag"], ["gh", "repo"],
                                      ["git", "push"], ["git", "push"], ["gh", "release"]]
    assert "--accept-visibility-change-consequences" in calls[3]


def test_publish_rejects_an_upload_without_a_commit_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(s7, "build_upload", lambda root, repo, out: "f" * 64)
    hf = FakeHF()
    hf.upload_folder = lambda **kw: type("CommitInfo", (), {"oid": "main"})()
    with pytest.raises(RuntimeError, match="no commit SHA"):
        s7.publish(tmp_path, "me/model", "v1.0.0", s7.Publisher(run=lambda cmd: pytest.fail(f"ran {cmd}"), hf=hf))


def test_model_card_has_license_and_results():
    card = s7.model_card("me/model", "| min cosine | 0.99 |")
    assert card.startswith("---\nlicense: apache-2.0\nbase_model: google/embeddinggemma-2\n")
    assert "| min cosine | 0.99 |" in card and '"me/model"' in card
