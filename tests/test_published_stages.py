"""s5, s6 and s7 on what s4 shipped (ADR-0010): both budget archives unpacked, sha256-checked, bundled and uploaded
per budget. The heavy parts (egcm-parity, Core ML, the device, the Hub) are replaced."""

import json

import pytest
from test_published import BUDGET_FUNCTIONS, fake_tokenizer, needs_aa, put_s4

from egcm import published
from egcm.report import Report, write_report
from stages.s5_swift import run as s5
from stages.s6_device import run as s6
from stages.s7_publish import run as s7


@needs_aa
def test_s5_runs_on_both_unpacked_archives_and_records_them(tmp_path, monkeypatch):
    d = put_s4(tmp_path)
    seen = {}

    def measure(model, out, tokenizer):
        seen.update(model=model, tokenizer=tokenizer)
        return {"m": 1}, {"d": 2}

    monkeypatch.setattr(s5.published, "shipped", lambda real=published.shipped: real(tmp_path))
    monkeypatch.setattr(s5.published, "check_tokenizer", lambda d: d)  # the fake tokenizer has other bytes
    monkeypatch.setattr(s5, "stage_dir", lambda _: tmp_path / "out")
    monkeypatch.setattr(s5, "measure", measure)
    r = Report(stage="s5_swift")
    s5.fill(r)
    unpacked = tmp_path / "out" / "unpacked"
    assert seen == {"model": {70: unpacked / "b70" / "EmbeddingGemma2.mlpackage",
                              280: unpacked / "b280" / "EmbeddingGemma2.mlpackage"},
                    "tokenizer": unpacked / "b70" / "tokenizer"}
    assert r.metrics == {"m": 1}
    assert r.details == {"d": 2, "s4_recipe": "int8_table4", "device_archives": {
        str(b): {"name": f"EmbeddingGemma2-b{b}.aar", "sha256": published.sha256_file(d / f"EmbeddingGemma2-b{b}.aar")}
        for b in (70, 280)}}


@needs_aa
def test_s6_bundles_one_layout_per_budget(tmp_path, monkeypatch):
    put_s4(tmp_path)
    monkeypatch.setattr(s6, "MODEL_DIR", tmp_path / "Model")
    (tmp_path / "Model").mkdir()
    (tmp_path / "Model" / "files.json").write_text("[]")  # a flat layout from an older run must not survive
    monkeypatch.setattr(s6.published, "shipped", lambda real=published.shipped: real(tmp_path))
    monkeypatch.setattr(s6.published, "check_tokenizer", lambda d: d)
    expected, details = s6.bundle(s6.argparse.Namespace(package=None, device=None), tmp_path / "out")
    assert s6.bundled_layouts() == ["b280", "b70"]
    for b in ("70", "280"):
        sha = published.sha256_file(tmp_path / "Model" / f"b{b}" / "files.json")
        assert expected[b] == "file-" + sha[:16] and details["files_json_sha256"][b] == sha
        listed = [e["path"] for e in json.loads((tmp_path / "Model" / f"b{b}" / "files.json").read_text())]
        assert "EmbeddingGemma2.mlpackage/Manifest.json" in listed
    assert expected["70"] != expected["280"] and details["s4_recipe"] == "int8_table4"
    unpacked = tmp_path / "out" / "unpacked"
    assert details["package"] == {b: str(unpacked / f"b{b}" / "EmbeddingGemma2.mlpackage") for b in ("70", "280")}


def test_s6_package_override_bundles_it_flat_for_both_budgets(tmp_path, monkeypatch):
    pkg = tmp_path / "EmbeddingGemma2.mlpackage"
    pkg.mkdir()
    (pkg / "Manifest.json").write_text("{}")
    monkeypatch.setattr(s6, "MODEL_DIR", tmp_path / "Model")
    monkeypatch.setattr(s6.published, "tokenizer_source", lambda: fake_tokenizer(tmp_path))
    expected, details = s6.bundle(s6.argparse.Namespace(package=str(pkg), device=None), tmp_path / "out")
    sha = published.sha256_file(tmp_path / "Model" / "files.json")
    assert expected == {"70": "file-" + sha[:16], "280": "file-" + sha[:16]}
    assert s6.bundled_layouts() == ["."] and details["s4_recipe"] is None


def test_s6_checks_each_budgets_revision_in_both_launches():
    want = {"70": "file-aaaa", "280": "file-bbbb"}
    s6.check_revisions({"model_revisions": want, "compile": {"model_revisions": want}}, want)
    one = {"70": "file-aaaa"}  # a device runs its own budget only
    s6.check_revisions({"model_revisions": one, "compile": {"model_revisions": one}}, want)
    with pytest.raises(RuntimeError, match="the benchmark ran models"):
        s6.check_revisions({"model_revisions": {"70": "file-bbbb"}, "compile": {"model_revisions": want}}, want)
    with pytest.raises(RuntimeError, match="the compile ran models None"):
        s6.check_revisions({"model_revisions": want, "compile": {"model_revision": "file-aaaa"}}, want)


def test_s7_preflight_names_the_missing_budget_files(tmp_path):
    pub = s7.Publisher(run=lambda cmd: s7.subprocess.CompletedProcess(cmd, 1, "", ""))
    d = put_s4(tmp_path, archive=False)
    (d / "checksums.json").write_text(json.dumps({"archive": {f"EmbeddingGemma2-b{b}.aar": "0" * 64
                                                              for b in (70, 280)}}))
    problems = s7.preflight(tmp_path, "v1.0.0", pub)
    assert any("EmbeddingGemma2-b280.aar (s4 recipe int8_table4, budget 280) is missing" in p for p in problems)

    put_s4(tmp_path, winner=None, archive=False)
    problems = s7.preflight(tmp_path, "v1.0.0", pub)
    assert any(p.startswith("s4_compress: s4 shipped nothing") for p in problems)


@needs_aa
def test_build_upload_has_both_budget_folders(tmp_path, monkeypatch):
    put_s4(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "RESULTS.md").write_text("| r |\n")
    (tmp_path / "NOTICE").write_text("n\n")
    (tmp_path / "LICENSE").write_text("apache\n")
    monkeypatch.setattr(s7.published, "tokenizer_source", lambda root=None: tmp_path / "src" / "snapshot")
    out = tmp_path / "upload"
    (out / "EmbeddingGemma2.mlpackage").mkdir(parents=True)  # a flat layout from an older upload is removed
    shas = s7.build_upload(tmp_path, "me/model", "v1.0.0", out)
    assert sorted(p.name for p in out.iterdir()) == ["LICENSE", "NOTICE", "README.md", "b280", "b70"]
    for b in ("70", "280"):
        f = out / f"b{b}"
        assert shas["manifest_json_sha256"][b] == published.sha256_file(f / "manifest.json")
        assert shas["files_json_sha256"][b] == published.sha256_file(f / "files.json")
        assert sorted(p.name for p in f.iterdir()) == [
            "EmbeddingGemma2.aar", "EmbeddingGemma2.mlpackage", "files.json", "manifest.json", "tokenizer"]
        listed = [e["path"] for e in json.loads((f / "files.json").read_text())]
        assert "EmbeddingGemma2.aar" not in listed and "manifest.json" not in listed  # the plain layout only
        assert json.loads((f / "manifest.json").read_text())["image_budget"] == int(b)
    card = (out / "README.md").read_text()
    assert "`b<budget>/EmbeddingGemma2.aar`" in card and "`b70/`" in card and "`b280/`" in card


@needs_aa
def test_model_card_names_each_budgets_archive_and_measured_sizes(tmp_path):
    put_s4(tmp_path)
    r = published.read_report(tmp_path / "artifacts" / "s4_compress" / "report.json")
    r.details["variants"] = {"int8_table4": {"download_mb": 358.7, "package_mb": 452.4}}
    r.details["budgets"]["70"] |= {"download_mb": 343.12, "package_mb": 430.0}
    r.details["budgets"]["280"] |= {"download_mb": 352.81, "package_mb": 440.0}
    r.metrics["max_download_mb"] = 352.81
    write_report(r, tmp_path / "artifacts" / "s4_compress")
    v = s7.card.values(tmp_path, {"70": {"unpacked_size": 470_000_000}, "280": {"unpacked_size": 484_000_000}})
    assert (v["archive"], v["manifest"], v["download_mb"], v["max_download_mb"]) == (
        "EmbeddingGemma2.aar", "manifest.json", "352.8", "360")
    assert (v["download_mb_70"], v["download_mb_280"], v["unpacked_mb_70"], v["unpacked_mb_280"]) == (
        "343.1", "352.8", "470.0", "484.0")
    card = s7.model_card("me/model", "| r |", v, tmp_path)
    assert "343.1 MB for budget 70, 352.8 MB for budget 280" in card and "the s4 gate is 360 MB" in card
    assert "358.7" not in card  # the 14-function archive is not what anyone downloads
    assert "zipped" not in card and "32 MB" not in card  # no hand-typed sizes left
    assert s7.card.values(tmp_path)["unpacked_mb_70"] == s7.card.PENDING
    assert {b: len(f) for b, f in BUDGET_FUNCTIONS.items()} == {70: 9, 280: 9}
