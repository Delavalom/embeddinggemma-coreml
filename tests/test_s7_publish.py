"""s7's guards, with fakes for every side effect: no network, no git writes, no Hugging Face calls."""

import subprocess

import pytest
from conftest import commit

from egcm.paths import STAGES
from egcm.provenance import code_hash, inputs_sha256
from egcm.report import Report, read_report, write_report
from stages.s7_publish import run as s7

SHA = "0123456789abcdef0123456789abcdef01234567"
SHAS = {"files_json_sha256": "f" * 64, "manifest_json_sha256": "e" * 64}


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
    monkeypatch.setattr(s7, "build_upload", lambda root, repo, tag, out: SHAS)
    hf = FakeHF()
    run, calls = fake_runner()
    details = s7.publish(tmp_path, "me/model", "v1.0.0", s7.Publisher(run=run, hf=hf))
    assert details == {"hf_repo": "me/model", "revision": SHA, **SHAS, "tag": "v1.0.0"}
    assert hf.calls[0] == ("create_repo", {"repo_id": "me/model", "repo_type": "model", "private": False,
                                           "exist_ok": True})
    assert hf.calls[1][0] == "upload_folder"
    swift = (tmp_path / s7.DEFAULT_MODEL_SWIFT).read_text()
    assert 'defaultRepo = "me/model"' in swift and f'defaultRevision = "{SHA}"' in swift
    assert f'defaultManifestSHA256: String? = "{"e" * 64}"' in swift
    assert [c[:2] for c in calls] == [["git", "add"], ["git", "commit"], ["git", "tag"], ["gh", "repo"],
                                      ["git", "push"], ["git", "push"], ["gh", "release"]]
    assert "--accept-visibility-change-consequences" in calls[3]


def test_publish_rejects_an_upload_without_a_commit_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(s7, "build_upload", lambda root, repo, tag, out: SHAS)
    hf = FakeHF()
    hf.upload_folder = lambda **kw: type("CommitInfo", (), {"oid": "main"})()
    with pytest.raises(RuntimeError, match="no commit SHA"):
        s7.publish(tmp_path, "me/model", "v1.0.0", s7.Publisher(run=lambda cmd: pytest.fail(f"ran {cmd}"), hf=hf))


def test_model_card_has_license_and_results(tmp_path):
    card = s7.model_card("me/model", "| min cosine | 0.99 |", root=tmp_path)
    assert card.startswith("---\nlicense: apache-2.0\nbase_model: google/embeddinggemma-2\n")
    assert "| min cosine | 0.99 |" in card and "`me/model`" in card and "huggingface.co/me/model" in card


def test_model_card_without_reports_renders_every_placeholder_as_pending(tmp_path):
    card = s7.model_card("me/model", "results", root=tmp_path)
    assert "$" not in card.replace("$$", "")  # Template.substitute raised if a placeholder had no value
    assert "Gemma Prohibited Use Policy" in card and "Apache License, Version 2.0" in card
    assert f"| {s7.card.PENDING}: filled from s6_device |" in card
    assert "| s0_reference | not run |" in card


def write(root, stage, **fields):
    d = root / "artifacts" / stage
    d.mkdir(parents=True, exist_ok=True)
    write_report(Report(stage=stage, status="passed", passed=True, **fields), d)


def test_model_card_numbers_come_from_the_reports(tmp_path):
    write(tmp_path, "s0_reference", git_sha="a" * 40, code_hash="c" * 64,
          details={"revision": "r" * 40, "license": "apache-2.0", "raw_vs_bucket_cosine": {"70": {"min": 0.9158}}})
    variant = {"description": "int8 convs", "download_mb": 349.698, "package_mb": 452.4,
               "min_cosine_text": 0.9979129, "min_cosine_image": 0.9913288,
               "vs_bucket": {"by_budget": {"70": {"recall_at_10": 0.96166}, "280": {"recall_at_10": 0.96583}}},
               "latency_cpu_and_ne": {"embed_text_16": {"median_ms": 4.424}},
               "by_function": {"image": {"embed_image_280_landscape": {"worst": [["i161", 0.99133], ["i150", 0.9943]]},
                                         "embed_image_70_wide": {"worst": [["i141", 0.99322]]}}}}
    write(tmp_path, "s4_compress", details={"chosen": "int8_table4", "variants": {"int8_table4": variant}})
    device = {"os_version": "26.0", "peak_rss_mb": 612.4, "query": {"ms": {"median": 7.0}},
              "budgets": {"70": {"ms_per_image": {"median": 41.25}}}}
    write(tmp_path, "s6_device", metrics={"ms_per_image_70_median": 41.25}, details={"devices": {"iPhone16,1": device}})
    v = s7.card.values(tmp_path)
    assert v["variant"] == "int8_table4" and v["download_mb"] == "349.7" and v["package_mb"] == "452.4"
    assert (v["text_cos"], v["image_cos"]) == ("0.99791", "0.99133")
    assert (v["recall_70"], v["recall_280"], v["ms_text_16"]) == ("0.962", "0.966", "4.4")
    assert v["base_revision"] == "r" * 40 and v["raw_vs_bucket_70"] == "0.916" and v["raw_vs_bucket_280"] == "pending"
    assert v["worst_images"].startswith("i161 0.99133 (`embed_image_280_landscape`), i141 0.99322")
    assert v["s6_devices"] == "| iPhone16,1 | 26.0 | 41.2 | pending | 7.0 | 612 |"
    assert "| s0_reference | passed | `aaaaaaaaaaaa` | `cccccccccccccccc` |" in v["provenance"]
    card = s7.model_card("me/model", "results", root=tmp_path)
    assert "**0.99791**" in card and "| iPhone16,1 | 26.0 |" in card


def test_upload_ships_license_and_notice(tmp_path):
    (tmp_path / "LICENSE").write_text("apache\n")
    (tmp_path / "NOTICE").write_text("notice\n")
    out = tmp_path / "upload"
    out.mkdir()
    s7.copy_extras(tmp_path, out)
    assert (out / "LICENSE").read_text() == "apache\n" and (out / "NOTICE").read_text() == "notice\n"


def test_release_notes_are_the_draft_then_results(tmp_path):
    (tmp_path / "docs" / "release-notes").mkdir(parents=True)
    (tmp_path / "docs" / "RESULTS.md").write_text("# Results\n")
    out = tmp_path / "artifacts" / "s7_publish" / "release_notes.md"
    assert s7.release_notes(tmp_path, "v1.0.0", out).read_text() == "# Results\n"
    (tmp_path / "docs" / "release-notes" / "v1.0.0.md").write_text("# v1.0.0\n\nFirst release.\n\n")
    assert s7.release_notes(tmp_path, "v1.0.0", out).read_text() == "# v1.0.0\n\nFirst release.\n\n# Results\n"
