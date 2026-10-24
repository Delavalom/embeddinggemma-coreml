"""egcm.parts: measurement helpers, and the resumable build with the child processes replaced by fakes."""

import json
import platform

import pytest

from egcm import parts

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="libproc / sysctl are macOS")


def test_measurements():
    big = bytearray(200_000_000)
    assert parts.footprint_gb() > 0.19
    del big
    assert parts.swap_used_gb() >= 0
    assert parts.disk_free_gb(parts.Path("/")) > 0


def test_tree_sha_tracks_content_and_names(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x").write_text("1")
    s = parts.tree_sha(tmp_path)
    (tmp_path / "a" / "x").write_text("2")
    assert parts.tree_sha(tmp_path) != s
    (tmp_path / "a" / "x").write_text("1")
    assert parts.tree_sha(tmp_path) == s


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Children that write small dummy parts and 'merge' by appending the function name to a file."""
    calls = []

    def run(args, tmp):
        calls.append(args[:3])
        a = dict(zip(args[1::2], args[2::2], strict=True))
        if args[0] == "convert":
            recipes = [r for r in a["--recipes"].split(",") if r] or [""]
            for r in recipes:
                d = parts.Path(a["--out"]) / r / f"{a['--function']}.mlpackage"
                d.mkdir(parents=True)
                (d / "w").write_text(a["--function"] * 100)
            return {"peak_footprint_gb": 1.0}
        into = parts.Path(a["--into"])
        into.mkdir(exist_ok=True)
        f = into / "functions"
        names = (f.read_text().split() if f.exists() else []) + [a["--function"]]
        f.write_text(" ".join(names))
        return {"functions": names, "peak_footprint_gb": 0.5}

    monkeypatch.setattr(parts, "_run", run)
    monkeypatch.setattr(parts, "_need_disk", lambda *a, **k: None)
    monkeypatch.setattr("egcm.provenance.code_hash", lambda stage: "h1")
    return calls, tmp_path


def test_build_merges_every_function_and_cleans_up(fake):
    calls, out = fake
    r = parts.build("s2_coreml_fp32", ["a", "b", "c"], "fp32", out, log=lambda _: None)
    assert r["packages"]["EmbeddingGemma2"]["functions"] == ["a", "b", "c"]
    assert [c[0] for c in calls] == ["convert", "merge"] * 3
    assert not (out / "parts").exists()
    assert set(r["peak_footprint_gb"]) == {"convert/a", "convert/b", "convert/c", "merge/EmbeddingGemma2/a",
                                           "merge/EmbeddingGemma2/b", "merge/EmbeddingGemma2/c"}


def test_rerun_skips_merged_functions(fake):
    calls, out = fake
    parts.build("s2_coreml_fp32", ["a", "b"], "fp32", out, log=lambda _: None)
    calls.clear()
    r = parts.build("s2_coreml_fp32", ["a", "b", "c"], "fp32", out, log=lambda _: None)
    assert calls == [["convert", "--function", "c"], ["merge", "--into", str(out / "EmbeddingGemma2.mlpackage")]]
    assert r["packages"]["EmbeddingGemma2"]["functions"] == ["a", "b", "c"]


def test_a_saved_part_is_reused_and_a_tampered_one_is_not(fake):
    calls, out = fake
    parts.build("s2_coreml_fp32", ["a"], "fp32", out, log=lambda _: None)
    # simulate a crash after converting b but before merging it: the part and its sha are on record
    state = json.loads((out / "build.json").read_text())
    d = out / "parts" / "b.mlpackage"
    d.mkdir(parents=True)
    (d / "w").write_text("b" * 100)
    state["parts"]["EmbeddingGemma2/b"] = {"mb": 0.0, "sha256": parts.tree_sha(d)}
    (out / "build.json").write_text(json.dumps(state))
    calls.clear()
    parts.build("s2_coreml_fp32", ["a", "b"], "fp32", out, log=lambda _: None)
    assert [c[0] for c in calls] == ["merge"]  # b's part was reused


def test_changed_code_or_tampered_package_starts_over(fake, monkeypatch):
    calls, out = fake
    parts.build("s2_coreml_fp32", ["a", "b"], "fp32", out, log=lambda _: None)
    (out / "EmbeddingGemma2.mlpackage" / "functions").write_text("tampered")
    calls.clear()
    parts.build("s2_coreml_fp32", ["a", "b"], "fp32", out, log=lambda _: None)
    assert [c[0] for c in calls] == ["convert", "merge"] * 2
    monkeypatch.setattr("egcm.provenance.code_hash", lambda stage: "h2")
    calls.clear()
    parts.build("s2_coreml_fp32", ["a", "b"], "fp32", out, log=lambda _: None)
    assert [c[0] for c in calls] == ["convert", "merge"] * 2


def test_recipes_build_one_package_each_from_one_conversion(fake):
    calls, out = fake
    r = parts.build("s4_compress", ["a", "b"], "fp16", out, recipes=["x", "y"], log=lambda _: None)
    assert [c[0] for c in calls] == ["convert", "merge", "merge"] * 2
    assert set(r["packages"]) == {"x", "y"}
    assert all(p["functions"] == ["a", "b"] for p in r["packages"].values())


def test_disk_guard_waits_then_fails(monkeypatch, tmp_path):
    free = iter([10.0, 10.0, 30.0])
    monkeypatch.setattr(parts, "disk_free_gb", lambda _: next(free))
    monkeypatch.setattr(parts.time, "sleep", lambda _: None)
    logs = []
    parts._need_disk(tmp_path, 2.0, "merge x", floor=15.0, log=logs.append)  # waits until 30 GB free
    assert len(logs) == 1 and "waiting" in logs[0]
    monkeypatch.setattr(parts, "disk_free_gb", lambda _: 10.0)
    with pytest.raises(RuntimeError, match=r"merge y needs about 2\.0 GB"):
        parts._need_disk(tmp_path, 2.0, "merge y", floor=15.0, log=logs.append, wait_s=0)
