import pytest
from conftest import commit

from egcm.provenance import (
    STAGE_CODE,
    STAGE_INPUTS,
    MissingInputError,
    code_hash,
    git_dirty,
    git_sha,
    inputs_sha256,
    is_stale,
    watched_paths,
)
from egcm.report import Report, write_report


def write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def put_report(root, stage, **kw):
    d = root / "artifacts" / stage
    d.mkdir(parents=True, exist_ok=True)
    write_report(Report(stage=stage, **kw), d)


def test_watched_paths():
    assert watched_paths("s0_reference") == [
        "stages/s0_reference", "egcm/buckets.py", "egcm/fixtures.py", "egcm/paths.py", "pyproject.toml", "uv.lock"]
    s1 = watched_paths("s1_rewrite")
    assert s1[0] == "stages/s1_rewrite" and "egcm/model" in s1 and "egcm/weights.py" in s1
    assert "swift" in watched_paths("s5_swift") and "Examples" in watched_paths("s6_device")


def test_every_stage_has_code_paths():
    assert set(STAGE_CODE) == set(STAGE_INPUTS)
    for stage, paths in STAGE_CODE.items():
        assert paths[0] == f"stages/{stage}"
        assert {"pyproject.toml", "uv.lock", "egcm/paths.py"} <= set(paths)
        assert not {"egcm", "egcm/stage.py", "egcm/provenance.py", "egcm/report.py", "egcm/gates.py", "scripts",
                    "gates.toml"} & set(paths), stage


def test_model_change_makes_s1_stale_but_not_s0(repo):
    s0 = code_hash("s0_reference", repo)
    s1 = code_hash("s1_rewrite", repo)
    write(repo, "egcm/model/x.py", "M = 2\n")
    assert code_hash("s0_reference", repo) == s0 and not git_dirty("s0_reference", repo)
    assert code_hash("s1_rewrite", repo) != s1 and git_dirty("s1_rewrite", repo)
    commit(repo)
    assert code_hash("s0_reference", repo) == s0


def test_infra_modules_do_not_change_any_hash(repo):
    before = {s: code_hash(s, repo) for s in STAGE_CODE}
    write(repo, "egcm/stage.py", "S = 2\n")
    write(repo, "scripts/gen_status.py", "# changed\n")
    commit(repo)
    assert {s: code_hash(s, repo) for s in STAGE_CODE} == before


def test_only_the_stage_gate_table_counts(repo):
    s0, s1 = code_hash("s0_reference", repo), code_hash("s1_rewrite", repo)
    write(repo, "gates.toml", "[s0_reference]\nmin_a = 0.5\n[s1_rewrite]\nmin_a = 0.7\n")
    assert code_hash("s0_reference", repo) == s0 and not git_dirty("s0_reference", repo)
    assert code_hash("s1_rewrite", repo) != s1 and git_dirty("s1_rewrite", repo)
    commit(repo)
    assert code_hash("s0_reference", repo) == s0
    # formatting and comments inside the table do not matter, only its values
    write(repo, "gates.toml", "# note\n[s0_reference]\nmin_a   = 0.5\n\n[s1_rewrite]\nmin_a = 0.7\n")
    assert code_hash("s0_reference", repo) == s0


def test_code_hash_clean_is_stable_and_runs_git_in_root(repo, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # not a git repo: git must run with cwd=root
    h = code_hash("s1_rewrite", repo)
    assert h == code_hash("s1_rewrite", repo)
    assert not git_dirty("s1_rewrite", repo)
    assert len(git_sha(repo)) == 40


@pytest.mark.parametrize(
    "rel", ["egcm/model/x.py", "egcm/paths.py", "stages/s1_rewrite/run.py", "pyproject.toml", "uv.lock"]
)
def test_code_hash_changes_with_watched_files(repo, rel):
    before = code_hash("s1_rewrite", repo)
    write(repo, rel, "# changed\n")
    assert git_dirty("s1_rewrite", repo)
    dirty = code_hash("s1_rewrite", repo)
    assert dirty != before
    commit(repo)
    assert code_hash("s1_rewrite", repo) not in (before, dirty)


def test_code_hash_ignores_unwatched_files(repo):
    before = code_hash("s1_rewrite", repo)
    write(repo, "README.md", "changed\n")
    write(repo, "stages/s0_reference/run.py", "# other stage\n")
    assert not git_dirty("s1_rewrite", repo)
    assert code_hash("s1_rewrite", repo) == before
    commit(repo)
    assert code_hash("s1_rewrite", repo) == before


def test_dirty_hash_tracks_working_tree_content(repo):
    write(repo, "egcm/model/x.py", "X = 2\n")
    a = code_hash("s1_rewrite", repo)
    write(repo, "egcm/model/x.py", "X = 3\n")
    assert code_hash("s1_rewrite", repo) != a


def test_untracked_new_file_counts(repo):
    before = code_hash("s1_rewrite", repo)
    write(repo, "egcm/model/new.py", "Y = 1\n")
    assert git_dirty("s1_rewrite", repo)
    a = code_hash("s1_rewrite", repo)
    assert a != before
    write(repo, "egcm/model/new.py", "Y = 2\n")
    assert code_hash("s1_rewrite", repo) != a


def test_missing_stage_dir_is_not_an_error(repo):
    assert code_hash("s2_coreml_fp32", repo)  # stages/s2_coreml_fp32 does not exist yet


def test_inputs_s0_depends_on_manifest_and_revision(repo):
    base = inputs_sha256("s0_reference", repo, hf_revision="r1")
    assert inputs_sha256("s0_reference", repo, hf_revision="r2") != base
    write(repo, "fixtures/MANIFEST.sha256", "def  img.jpg\n")
    assert inputs_sha256("s0_reference", repo, hf_revision="r1") != base


def test_inputs_missing_manifest_raises(repo):
    (repo / "fixtures" / "MANIFEST.sha256").unlink()
    with pytest.raises(MissingInputError):
        inputs_sha256("s0_reference", repo)


def test_inputs_follow_upstream_reports(repo):
    with pytest.raises(MissingInputError):
        inputs_sha256("s1_rewrite", repo)
    put_report(repo, "s0_reference", code_hash="c1", inputs_sha256="i1", status="passed", passed=True)
    a = inputs_sha256("s1_rewrite", repo)
    put_report(repo, "s0_reference", code_hash="c2", inputs_sha256="i1", status="passed", passed=True)
    b = inputs_sha256("s1_rewrite", repo)
    put_report(repo, "s0_reference", code_hash="c2", inputs_sha256="i2", status="passed", passed=True)
    c = inputs_sha256("s1_rewrite", repo)
    assert len({a, b, c}) == 3


def stamped(repo, stage, **kw):
    return Report(stage=stage, code_hash=code_hash(stage, repo), inputs_sha256=inputs_sha256(stage, repo), **kw)


def test_is_stale_rules(repo):
    s0 = stamped(repo, "s0_reference", status="passed", passed=True)
    put_report(repo, "s0_reference", **s0.model_dump(exclude={"stage"}))
    s1 = stamped(repo, "s1_rewrite", status="passed", passed=True)
    assert not is_stale(s0, repo) and not is_stale(s1, repo)

    # a change in code s0 uses makes both stale (s1 through the upstream check)
    write(repo, "egcm/paths.py", "X = 2\n")
    assert is_stale(s0, repo) and is_stale(s1, repo)
    write(repo, "egcm/paths.py", "X = 1\n")
    assert not is_stale(s1, repo)

    # a change in code only s1 uses makes s1 stale, not s0
    write(repo, "egcm/model/x.py", "M = 2\n")
    assert is_stale(s1, repo) and not is_stale(s0, repo)
    write(repo, "egcm/model/x.py", "M = 1\n")

    # upstream rerun with different code makes the downstream stale, but not upstream itself
    put_report(repo, "s0_reference", **{**s0.model_dump(exclude={"stage"}), "code_hash": "other"})
    assert is_stale(s1, repo) and not is_stale(s0, repo)

    # missing upstream report -> stale
    (repo / "artifacts" / "s0_reference" / "report.json").unlink()
    assert is_stale(s1, repo)


def test_is_stale_unknown_stage(repo):
    with pytest.raises(ValueError):
        is_stale(Report(stage="s9_nope"), repo)
