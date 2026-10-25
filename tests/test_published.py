"""s5, s6 and s7 take what s4 shipped from its report (package, device archive, recipe; never a hard-coded name), and
s7's manifest.json (egcm.manifest: the contract the Swift ModelStore and Roll read)."""

import hashlib
import json
import os
import shutil
import subprocess

import pytest

from egcm import manifest, published
from egcm.archive import device_archive
from egcm.report import Report, write_report
from stages.s5_swift import run as s5
from stages.s6_device import run as s6
from stages.s7_publish import run as s7

FUNCTIONS = list(manifest.REQUIRED_FUNCTIONS)
needs_aa = pytest.mark.skipif(shutil.which("aa") is None, reason="needs macOS `aa`")


def fake_sources(tmp_path):
    pkg = tmp_path / "src" / "int8_table4.mlpackage"
    (pkg / "Data" / "com.apple.CoreML" / "weights").mkdir(parents=True)
    (pkg / "Manifest.json").write_text("{}")
    (pkg / "Data" / "com.apple.CoreML" / "weights" / "weight.bin").write_bytes(os.urandom(50_000))
    tok = tmp_path / "src" / "snapshot"
    tok.mkdir()
    for name in [*published.TOKENIZER_FILES, "model.safetensors"]:  # only the two tokenizer files are archived
        (tok / name).write_text(name)
    return pkg, tok


def put_s4(root, winner="int8_table4", archive=True, checksums=True, functions=FUNCTIONS):
    """An s4 report and folder like the stage writes them; with `archive`, a real device archive (needs `aa`)."""
    d = root / "artifacts" / "s4_compress"
    d.mkdir(parents=True, exist_ok=True)
    details = {"chosen": winner or "closest", "winner": winner, "package": "EmbeddingGemma2.mlpackage",
               "device_archive": "EmbeddingGemma2.aar" if winner else None,
               "build": {"packages": {winner or "closest": {"functions": list(functions)[::-1]}}}}
    if archive:
        pkg, tok = fake_sources(root)
        shutil.copytree(pkg, d / "EmbeddingGemma2.mlpackage")
        device_archive(d / "EmbeddingGemma2.mlpackage", tok, d / "EmbeddingGemma2.aar")
        if checksums:
            (d / "checksums.json").write_text(json.dumps({"archive": {
                "EmbeddingGemma2.aar": published.sha256_file(d / "EmbeddingGemma2.aar"),
                "bytes": (d / "EmbeddingGemma2.aar").stat().st_size}}))
    write_report(Report(stage="s4_compress", status="passed" if winner else "failed", passed=bool(winner),
                        metrics={"max_download_mb": 350.2}, gate_values={"max_download_mb": 360}, details=details), d)
    return d


@needs_aa
def test_shipped_reads_the_report_and_checksums(tmp_path):
    d = put_s4(tmp_path)
    s4 = published.shipped(tmp_path)
    assert s4 == published.Shipped(package=d / "EmbeddingGemma2.mlpackage", archive=d / "EmbeddingGemma2.aar",
                                   archive_sha256=published.sha256_file(d / "EmbeddingGemma2.aar"),
                                   recipe="int8_table4", download_mb=350.2, max_download_mb=360,
                                   functions=tuple(sorted(FUNCTIONS)))


def test_shipped_refuses_what_s4_did_not_ship(tmp_path):
    with pytest.raises(FileNotFoundError, match="just stage 4"):
        published.shipped(tmp_path)
    put_s4(tmp_path, winner=None, archive=False)  # a failed s4 still leaves its closest variant in place
    with pytest.raises(RuntimeError, match="shipped nothing"):
        published.shipped(tmp_path)
    put_s4(tmp_path, archive=False)
    with pytest.raises(RuntimeError, match=r"checksums\.json"):
        published.shipped(tmp_path)


def test_required_functions_are_the_contract_names():
    assert manifest.REQUIRED_FUNCTIONS[:4] == ("embed_text_16", "embed_text_32", "embed_text_64", "embed_text_128")
    assert {f"embed_image_{b}_{s}" for b in (70, 280) for s in ("tall", "portrait", "square", "landscape", "wide")} \
        == set(manifest.REQUIRED_FUNCTIONS[4:])
    with pytest.raises(ValueError, match="embed_text_128"):
        manifest.check_functions([f for f in FUNCTIONS if f != "embed_text_128"])


@needs_aa
def test_unpack_checks_sha256_and_layout(tmp_path):
    d = put_s4(tmp_path)
    aar = d / "EmbeddingGemma2.aar"
    with pytest.raises(ValueError, match="sha256"):
        manifest.unpack(aar, tmp_path / "x", "0" * 64)
    x = manifest.unpack(aar, tmp_path / "x", published.sha256_file(aar))
    assert sorted(p.relative_to(x).as_posix() for p in x.rglob("*") if p.is_file()) == [
        "EmbeddingGemma2.mlpackage/Data/com.apple.CoreML/weights/weight.bin", "EmbeddingGemma2.mlpackage/Manifest.json",
        "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]


@needs_aa
@pytest.mark.parametrize("bad", ["extra_root_file", "symlink", "no_tokenizer"])
def test_unpack_rejects_other_layouts(tmp_path, bad):
    root = tmp_path / "root"
    (root / "EmbeddingGemma2.mlpackage").mkdir(parents=True)
    (root / "EmbeddingGemma2.mlpackage" / "Manifest.json").write_text("{}")
    if bad != "no_tokenizer":
        (root / "tokenizer").mkdir()
        for name in published.TOKENIZER_FILES:
            (root / "tokenizer" / name).write_text("t")
    if bad == "extra_root_file":
        (root / "README.md").write_text("x")
    if bad == "symlink":
        (root / "EmbeddingGemma2.mlpackage" / "link").symlink_to("/etc/hosts")
    aar = tmp_path / "bad.aar"
    subprocess.run(["aa", "archive", "-a", "lzfse", "-d", str(root), "-o", str(aar)], check=True)
    with pytest.raises(ValueError):
        manifest.unpack(aar, tmp_path / "x")
    assert not (tmp_path / "x").exists()


@needs_aa
def test_manifest_publishes_the_gated_archive(tmp_path):
    put_s4(tmp_path, functions=[*FUNCTIONS, "extra_fn"])
    s4, out = published.shipped(tmp_path), tmp_path / "upload"
    sha = manifest.write(s4, out, revision="v1.0.0")
    data = (out / "manifest.json").read_bytes()
    assert sha == hashlib.sha256(data).hexdigest()
    m = json.loads(data)
    aar = out / m["archive"]
    assert m["format"] == 1 and m["archive"] == "EmbeddingGemma2.aar" and "/" not in m["archive"]
    assert m["size"] == aar.stat().st_size and m["sha256"] == s4.archive_sha256 == published.sha256_file(aar)
    assert m["revision"] == "v1.0.0" and m["functions"] == sorted([*FUNCTIONS, "extra_fn"])
    assert m["unpacked_size"] == 2 + 50_000 + len("tokenizer.json") + len("tokenizer_config.json")

    s4.archive.write_bytes(s4.archive.read_bytes()[:-1])  # not the bytes s4 gated
    with pytest.raises(ValueError, match=r"checksums\.json"):
        manifest.write(s4, tmp_path / "again")


@needs_aa
def test_s5_runs_on_the_unpacked_archive_and_records_it(tmp_path, monkeypatch):
    put_s4(tmp_path)
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
    assert seen == {"model": tmp_path / "out" / "unpacked" / "EmbeddingGemma2.mlpackage",
                    "tokenizer": tmp_path / "out" / "unpacked" / "tokenizer"}
    assert r.metrics == {"m": 1}
    assert r.details == {"d": 2, "s4_recipe": "int8_table4", "device_archive": "EmbeddingGemma2.aar",
                         "device_archive_sha256": published.sha256_file(
                             tmp_path / "artifacts" / "s4_compress" / "EmbeddingGemma2.aar")}


@needs_aa
def test_s6_bundles_the_unpacked_archive(tmp_path, monkeypatch):
    put_s4(tmp_path)

    class Bundled(Exception):
        pass

    def assemble(package, tokenizer, out):
        raise Bundled(package, tokenizer)

    monkeypatch.setattr(s6.published, "shipped", lambda real=published.shipped: real(tmp_path))
    monkeypatch.setattr(s6.published, "check_tokenizer", lambda d: d)
    monkeypatch.setattr(s6.published, "assemble", assemble)
    monkeypatch.setattr(s6, "stage_dir", lambda _: tmp_path / "out")
    with pytest.raises(Bundled) as e:
        s6.make_fill(s6.argparse.Namespace(package=None, device=None))(Report(stage="s6_device"))
    unpacked = tmp_path / "out" / "unpacked"
    assert e.value.args == (unpacked / "EmbeddingGemma2.mlpackage", unpacked / "tokenizer")


def test_s7_preflight_names_the_shipped_recipe(tmp_path):
    pub = s7.Publisher(run=lambda cmd: s7.subprocess.CompletedProcess(cmd, 1, "", ""))
    d = put_s4(tmp_path, archive=False)
    (d / "checksums.json").write_text(json.dumps({"archive": {"EmbeddingGemma2.aar": "0" * 64}}))
    problems = s7.preflight(tmp_path, "v1.0.0", pub)
    assert any("EmbeddingGemma2.aar (s4 recipe int8_table4) is missing" in p for p in problems)

    put_s4(tmp_path, winner=None, archive=False)
    problems = s7.preflight(tmp_path, "v1.0.0", pub)
    assert any(p.startswith("s4_compress: s4 shipped nothing") for p in problems)


@needs_aa
def test_build_upload_has_the_archive_and_the_plain_layout(tmp_path, monkeypatch):
    put_s4(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "RESULTS.md").write_text("| r |\n")
    (tmp_path / "NOTICE").write_text("n\n")
    (tmp_path / "LICENSE").write_text("apache\n")
    monkeypatch.setattr(s7.published, "tokenizer_source", lambda root=None: tmp_path / "src" / "snapshot")
    out = tmp_path / "upload"
    shas = s7.build_upload(tmp_path, "me/model", "v1.0.0", out)
    assert shas["manifest_json_sha256"] == published.sha256_file(out / "manifest.json")
    assert shas["files_json_sha256"] == published.sha256_file(out / "files.json")
    listed = [e["path"] for e in json.loads((out / "files.json").read_text())]
    assert "EmbeddingGemma2.aar" not in listed and "README.md" not in listed  # files.json: the plain layout only
    names = {p.name for p in out.iterdir()}
    assert {"manifest.json", "EmbeddingGemma2.aar", "README.md", "NOTICE", "LICENSE", "files.json"} <= names
    assert "`EmbeddingGemma2.aar`" in (out / "README.md").read_text()


@needs_aa
def test_model_card_names_the_archive_and_its_measured_sizes(tmp_path):
    put_s4(tmp_path)
    r = published.read_report(tmp_path / "artifacts" / "s4_compress" / "report.json")
    r.details["variants"] = {"int8_table4": {"download_mb": 349.698, "package_mb": 452.4}}
    write_report(r, tmp_path / "artifacts" / "s4_compress")
    v = s7.card.values(tmp_path, {"unpacked_size": 484_000_000})
    assert (v["archive"], v["manifest"], v["download_mb"], v["max_download_mb"], v["unpacked_mb"]) == (
        "EmbeddingGemma2.aar", "manifest.json", "349.7", "360", "484.0")
    card = s7.model_card("me/model", "| r |", v, tmp_path)
    assert "`EmbeddingGemma2.aar` (349.7 MB" in card and "the s4 gate is 360 MB" in card
    assert "zipped" not in card and "32 MB" not in card  # no hand-typed sizes left
    assert s7.card.values(tmp_path)["unpacked_mb"] == s7.card.PENDING
