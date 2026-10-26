"""s5, s6 and s7 take what s4 shipped from its report (one package and device archive per image budget, ADR-0010;
never a hard-coded name), and s7's per-budget manifest.json (egcm.manifest: the contract the Swift ModelStore and
Roll read). The stages' use of it: tests/test_published_stages.py."""

import hashlib
import json
import os
import shutil
import subprocess

import pytest

from egcm import manifest, published
from egcm.archive import device_archive
from egcm.report import Report, write_report

FUNCTIONS = list(manifest.REQUIRED_FUNCTIONS)
BUDGET_FUNCTIONS = {b: list(manifest.required_functions(b)) for b in (70, 280)}
needs_aa = pytest.mark.skipif(shutil.which("aa") is None, reason="needs macOS `aa`")


def fake_package(path, payload=50_000):
    (path / "Data" / "com.apple.CoreML" / "weights").mkdir(parents=True)
    (path / "Manifest.json").write_text("{}")
    (path / "Data" / "com.apple.CoreML" / "weights" / "weight.bin").write_bytes(os.urandom(payload))
    return path


def fake_tokenizer(root):
    tok = root / "src" / "snapshot"
    tok.mkdir(parents=True, exist_ok=True)
    for name in [*published.TOKENIZER_FILES, "model.safetensors"]:  # only the two tokenizer files are archived
        (tok / name).write_text(name)
    return tok


def put_s4(root, winner="int8_table4", archive=True, checksums=True, functions=BUDGET_FUNCTIONS):
    """An s4 report and folder like the stage writes them (ADR-0010: the 14-function package plus one package and
    archive per budget); with `archive`, real device archives (needs `aa`)."""
    d = root / "artifacts" / "s4_compress"
    d.mkdir(parents=True, exist_ok=True)
    budgets = {str(b): {"package": f"EmbeddingGemma2-b{b}.mlpackage", "device_archive": f"EmbeddingGemma2-b{b}.aar",
                        "functions": list(fns)[::-1], "download_mb": 340.0 + b / 28, "package_mb": 440.0}
               for b, fns in functions.items()} if winner else None
    details = {"chosen": winner or "closest", "winner": winner, "package": "EmbeddingGemma2.mlpackage",
               "build": {"packages": {winner or "closest": {"functions": FUNCTIONS[::-1]}}}}
    if budgets:
        details["budgets"] = budgets
    if archive:
        tok = fake_tokenizer(root)
        fake_package(d / "EmbeddingGemma2.mlpackage")
        sums = {}
        for b, e in budgets.items():
            fake_package(d / e["package"], 50_000 + int(b))
            device_archive(d / e["package"], tok, d / e["device_archive"])
            sums[e["device_archive"]] = published.sha256_file(d / e["device_archive"])
        if checksums:
            (d / "checksums.json").write_text(json.dumps({"archive": sums}))
    write_report(Report(stage="s4_compress", status="passed" if winner else "failed", passed=bool(winner),
                        metrics={"max_download_mb": 350.2}, gate_values={"max_download_mb": 360}, details=details), d)
    return d


@needs_aa
def test_shipped_reads_the_report_and_checksums(tmp_path):
    d = put_s4(tmp_path)
    s4 = published.shipped(tmp_path)
    assert (s4.package, s4.recipe, s4.download_mb, s4.max_download_mb, s4.functions) == (
        d / "EmbeddingGemma2.mlpackage", "int8_table4", 350.2, 360, tuple(sorted(FUNCTIONS)))
    assert sorted(s4.budgets) == [70, 280]
    for b, v in s4.budgets.items():
        assert v == published.Variant(budget=b, package=d / f"EmbeddingGemma2-b{b}.mlpackage",
                                      archive=d / f"EmbeddingGemma2-b{b}.aar",
                                      archive_sha256=published.sha256_file(d / f"EmbeddingGemma2-b{b}.aar"),
                                      functions=tuple(sorted(BUDGET_FUNCTIONS[b])), download_mb=340.0 + b / 28)


def test_shipped_refuses_what_s4_did_not_ship(tmp_path):
    with pytest.raises(FileNotFoundError, match="just stage 4"):
        published.shipped(tmp_path)
    put_s4(tmp_path, winner=None, archive=False)  # a failed s4 still leaves its closest variant in place
    with pytest.raises(RuntimeError, match="shipped nothing"):
        published.shipped(tmp_path)
    put_s4(tmp_path, archive=False)
    with pytest.raises(RuntimeError, match=r"checksums\.json"):
        published.shipped(tmp_path)


def test_shipped_refuses_a_report_from_before_the_split(tmp_path):
    put_s4(tmp_path, archive=False, functions={70: BUDGET_FUNCTIONS[70]})
    with pytest.raises(RuntimeError, match=r"image budget\(s\) \[280\] \(made before ADR-0010\)"):
        published.shipped(tmp_path)


def test_required_functions_are_the_contract_names():
    assert manifest.REQUIRED_FUNCTIONS[:4] == ("embed_text_16", "embed_text_32", "embed_text_64", "embed_text_128")
    images = {b: {f"embed_image_{b}_{s}" for s in ("tall", "portrait", "square", "landscape", "wide")}
              for b in (70, 280)}
    assert images[70] | images[280] == set(manifest.REQUIRED_FUNCTIONS[4:])
    for b in (70, 280):
        assert set(manifest.required_functions(b)) == set(manifest.REQUIRED_FUNCTIONS[:4]) | images[b]
    with pytest.raises(ValueError, match="image budget"):
        manifest.required_functions(140)
    with pytest.raises(ValueError, match="embed_text_128"):
        manifest.check_functions([f for f in FUNCTIONS if f != "embed_text_128"])


def test_check_functions_per_budget():
    assert manifest.check_functions(BUDGET_FUNCTIONS[70], 70) == sorted(BUDGET_FUNCTIONS[70])
    assert manifest.check_functions([*BUDGET_FUNCTIONS[280], "extra"], 280) == sorted([*BUDGET_FUNCTIONS[280], "extra"])
    with pytest.raises(ValueError, match="embed_image_70_wide"):
        manifest.check_functions(BUDGET_FUNCTIONS[280], 70)
    with pytest.raises(ValueError, match="another budget's image functions"):
        manifest.check_functions(FUNCTIONS, 70)  # the 14-function package is not a budget package
    assert manifest.check_functions(FUNCTIONS) == sorted(FUNCTIONS)  # without a budget, all 14, as before


@needs_aa
def test_unpack_checks_sha256_and_layout(tmp_path):
    d = put_s4(tmp_path)
    aar = d / "EmbeddingGemma2-b70.aar"
    with pytest.raises(ValueError, match="sha256"):
        manifest.unpack(aar, tmp_path / "x", "0" * 64)
    x = manifest.unpack(aar, tmp_path / "x", published.sha256_file(aar))
    assert sorted(p.relative_to(x).as_posix() for p in x.rglob("*") if p.is_file()) == [
        "EmbeddingGemma2.mlpackage/Data/com.apple.CoreML/weights/weight.bin", "EmbeddingGemma2.mlpackage/Manifest.json",
        "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]  # the package is EmbeddingGemma2 inside


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
def test_manifest_publishes_each_budgets_gated_archive(tmp_path):
    put_s4(tmp_path, functions={70: [*BUDGET_FUNCTIONS[70], "extra_fn"], 280: BUDGET_FUNCTIONS[280]})
    s4, out = published.shipped(tmp_path), tmp_path / "upload"
    for b, v in s4.budgets.items():
        sha = manifest.write(v, out, revision="v1.0.0")
        data = (out / f"b{b}" / "manifest.json").read_bytes()
        assert sha == hashlib.sha256(data).hexdigest()
        m = json.loads(data)
        aar = out / f"b{b}" / m["archive"]
        assert m["format"] == 1 and m["archive"] == "EmbeddingGemma2.aar" and "/" not in m["archive"]
        assert m["image_budget"] == b and m["revision"] == "v1.0.0"
        assert m["size"] == aar.stat().st_size and m["sha256"] == v.archive_sha256 == published.sha256_file(aar)
        assert m["functions"] == sorted([*BUDGET_FUNCTIONS[b], *(["extra_fn"] if b == 70 else [])])
        assert m["unpacked_size"] == 2 + 50_000 + b + len("tokenizer.json") + len("tokenizer_config.json")
    assert sorted(p.name for p in out.iterdir()) == ["b280", "b70"]

    v = s4.budgets[280]
    v.archive.write_bytes(v.archive.read_bytes()[:-1])  # not the bytes s4 gated
    with pytest.raises(ValueError, match=r"checksums\.json"):
        manifest.write(v, tmp_path / "again")
    wrong = published.Variant(**{**v.__dict__, "functions": tuple(BUDGET_FUNCTIONS[70])})
    with pytest.raises(ValueError, match="lacks the functions"):
        manifest.write(wrong, tmp_path / "again")
