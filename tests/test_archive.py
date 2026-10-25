"""egcm.archive.device_archive: the device download's layout contract, determinism and round trip (needs `aa`)."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from egcm import archive

pytestmark = pytest.mark.skipif(not Path(archive.AA).exists(), reason="aa (Apple Archive) not available")


def package(root: Path, name: str = "int8_table4.mlpackage", order=("b", "a")) -> Path:
    p = root / name
    (p / "Data" / "com.apple.CoreML" / "weights").mkdir(parents=True)
    for n in order:  # created in a different order per test
        (p / "Data" / "com.apple.CoreML" / "weights" / f"{n}.bin").write_bytes(n.encode() * 5000)
    (p / "Data" / "com.apple.CoreML" / "model.mlmodel").write_bytes(b"spec" * 100)
    (p / "Manifest.json").write_text("{}")
    return p


def tokenizer(root: Path, extra: bool = True) -> Path:
    t = root / "tok"
    t.mkdir(parents=True)
    (t / "tokenizer.json").write_text(json.dumps({"model": "x" * 1000}))
    (t / "tokenizer_config.json").write_text("{}")
    if extra:  # files the device does not need stay out
        (t / "tokenizer.model").write_bytes(b"sentencepiece")
        (t / "config.json").write_text("{}")
    return t


def entries(aar: Path) -> list[str]:
    out = subprocess.run([archive.AA, "list", "-i", str(aar)], check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line]


def test_layout_is_exactly_the_contract(tmp_path):
    aar = archive.device_archive(package(tmp_path), tokenizer(tmp_path), tmp_path / "out" / "M.aar")
    assert aar == tmp_path / "out" / "M.aar" and aar.is_file()
    assert entries(aar) == ["EmbeddingGemma2.mlpackage", "EmbeddingGemma2.mlpackage/Data",
                            "EmbeddingGemma2.mlpackage/Data/com.apple.CoreML",
                            "EmbeddingGemma2.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                            "EmbeddingGemma2.mlpackage/Data/com.apple.CoreML/weights",
                            "EmbeddingGemma2.mlpackage/Data/com.apple.CoreML/weights/a.bin",
                            "EmbeddingGemma2.mlpackage/Data/com.apple.CoreML/weights/b.bin",
                            "EmbeddingGemma2.mlpackage/Manifest.json",
                            "tokenizer", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
    assert all(not e.startswith("/") and ".." not in Path(e).parts for e in entries(aar))
    assert entries(aar) == sorted(entries(aar))
    assert not list((tmp_path / "out").glob(".device_archive_*"))  # staging removed


def test_same_inputs_same_bytes(tmp_path):
    a = archive.device_archive(package(tmp_path / "a", order=("a", "b")), tokenizer(tmp_path / "a"), tmp_path / "a.aar")
    b = archive.device_archive(package(tmp_path / "b", order=("b", "a")), tokenizer(tmp_path / "b", extra=False),
                               tmp_path / "b.aar")
    sha = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (a, b)]
    assert sha[0] == sha[1]


def test_lzfse_and_round_trip(tmp_path):
    pkg, tok = package(tmp_path), tokenizer(tmp_path)
    aar = archive.device_archive(pkg, tok, tmp_path / "M.aar")
    assert aar.read_bytes()[:4] == b"pbze"  # LZFSE-compressed Apple Archive stream
    dest = tmp_path / "x"
    dest.mkdir()
    subprocess.run([archive.AA, "extract", "-i", str(aar), "-d", str(dest)], check=True, capture_output=True)
    got = dest / "EmbeddingGemma2.mlpackage" / "Data" / "com.apple.CoreML" / "weights" / "a.bin"
    assert got.read_bytes() == (pkg / "Data" / "com.apple.CoreML" / "weights" / "a.bin").read_bytes()
    assert (dest / "tokenizer" / "tokenizer.json").read_text() == (tok / "tokenizer.json").read_text()
    assert not any(p.is_symlink() for p in dest.rglob("*"))


def test_symlinked_tokenizer_source_becomes_a_regular_file(tmp_path):
    tok = tokenizer(tmp_path / "blobs", extra=False)
    linked = tmp_path / "snapshot"
    linked.mkdir()
    for n in archive.TOKENIZER_FILES:
        (linked / n).symlink_to(tok / n)  # the Hugging Face cache layout
    aar = archive.device_archive(package(tmp_path), linked, tmp_path / "M.aar")
    dest = tmp_path / "x"
    dest.mkdir()
    subprocess.run([archive.AA, "extract", "-i", str(aar), "-d", str(dest)], check=True, capture_output=True)
    assert not (dest / "tokenizer" / "tokenizer.json").is_symlink()
    assert (dest / "tokenizer" / "tokenizer.json").read_text() == (tok / "tokenizer.json").read_text()


def test_rejects_bad_inputs(tmp_path):
    pkg, tok = package(tmp_path), tokenizer(tmp_path)
    (tok / "tokenizer_config.json").unlink()
    with pytest.raises(FileNotFoundError):
        archive.device_archive(pkg, tok, tmp_path / "M.aar")
    (pkg / "link").symlink_to(pkg / "Manifest.json")
    with pytest.raises(ValueError):
        archive.stage_files(pkg, tokenizer(tmp_path / "t2"))
    with pytest.raises(FileNotFoundError):
        archive.stage_files(tmp_path / "tok", tok)  # not a package
