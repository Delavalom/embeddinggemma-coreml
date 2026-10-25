"""The published model folder: the layout `ModelStore` installs from (a Hugging Face repo or a local folder).

    files.json                               every file below, with size and sha256 (FilesManifest in Swift)
    EmbeddingGemma2.mlpackage/**             the s4 package, uncompressed
    tokenizer/tokenizer.json                 from google/embeddinggemma-2 at HF_REVISION
    tokenizer/tokenizer_config.json

s6 builds it into the demo app's bundle; s7 uploads it. Which s4 package that is comes from s4's report
(`shipped`), never from a recipe name written here. `files.json` matches
swift/Sources/EmbeddingGemma/Checksums.swift: a JSON array of {path, size, sha256}, sorted by path, covering
every regular file except `files.json` itself and hidden files (any path component starting with ".").
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from egcm.paths import ARTIFACTS, HF_REPO, HF_REVISION
from egcm.report import read_report

PACKAGE_NAME = "EmbeddingGemma2.mlpackage"
TOKENIZER_DIR = "tokenizer"
FILES_JSON = "files.json"
# sha256 of the tokenizer files at HF_REVISION (their Git LFS oids; same values as swift/Tests/.../TestSupport.swift).
TOKENIZER_FILES = {
    "tokenizer.json": "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    "tokenizer_config.json": "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874",
}


S4_STAGE = "s4_compress"


@dataclass(frozen=True)
class Shipped:
    """What s4 ships, as its report records it."""

    package: Path  # artifacts/s4_compress/<details.package>
    recipe: str  # details.winner: the first egcm.compress.RECIPES entry that passed every s4 gate (ADR-0007)
    download_mb: float | None  # the zipped package (metric max_download_mb)
    max_download_mb: float | None  # the gate it was checked against (gates.toml [s4_compress], as recorded)


def shipped(root: Path | None = None) -> Shipped:
    """The package s4 shipped, read from artifacts/s4_compress/report.json.

    s4 renames its winning recipe to `details.package`. A failed s4 still leaves its closest variant there, so a
    report without `details.winner` is refused rather than trusted."""
    artifacts = ARTIFACTS if root is None else Path(root) / "artifacts"
    path = artifacts / S4_STAGE / "report.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; run `just stage 4`")
    r = read_report(path)
    winner, name = r.details.get("winner"), r.details.get("package")
    if not winner or not name:
        raise RuntimeError(f"s4 shipped nothing (status {r.status!r}, winner {winner!r}); rerun `just stage 4`")
    return Shipped(package=artifacts / S4_STAGE / name, recipe=winner, download_mb=r.metrics.get("max_download_mb"),
                   max_download_mb=r.gate_values.get("max_download_mb"))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def files_manifest(root: Path) -> list[dict]:
    """Every regular file under `root` except files.json and hidden files, sorted by path."""
    root = Path(root).resolve()
    out = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts) or not p.is_file() or rel.as_posix() == FILES_JSON:
            continue
        out.append({"path": rel.as_posix(), "size": p.stat().st_size, "sha256": sha256_file(p)})
    return sorted(out, key=lambda e: e["path"])


def write_files_manifest(root: Path) -> str:
    """Writes root/files.json and returns its sha256 (what `defaultFilesManifestSHA256` pins)."""
    data = (json.dumps(files_manifest(root), indent=2, sort_keys=True) + "\n").encode()
    (Path(root) / FILES_JSON).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def tokenizer_source(root: Path | None = None) -> Path:
    """A folder with both tokenizer files at HF_REVISION, verified by sha256.

    Looks in s0's snapshot (artifacts/hf/...) and in the Swift tests' cache (artifacts/tokenizer/<revision>/)."""
    artifacts = ARTIFACTS if root is None else Path(root) / "artifacts"
    candidates = [
        artifacts / "hf" / f"models--{HF_REPO.replace('/', '--')}" / "snapshots" / HF_REVISION,
        artifacts / "tokenizer" / HF_REVISION,
    ]
    for d in candidates:
        if all((d / name).is_file() for name in TOKENIZER_FILES):
            bad = [n for n, sha in TOKENIZER_FILES.items() if sha256_file(d / n) != sha]
            if bad:
                raise RuntimeError(f"tokenizer files in {d} do not match HF_REVISION: {bad}")
            return d
    raise FileNotFoundError(
        "no tokenizer files found; run `just stage 0` (artifacts/hf) or `swift test` in swift/ "
        f"(artifacts/tokenizer). Looked in: {', '.join(str(c) for c in candidates)}"
    )


def _clone_tree(src: Path, dst: Path) -> None:
    """APFS clone (no extra disk space) where possible, else a normal copy."""
    if subprocess.run(["cp", "-cR", str(src), str(dst)], capture_output=True).returncode != 0:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)


def assemble(package: Path, tokenizer: Path, out: Path) -> str:
    """Builds the published folder at `out` (replacing it) and returns the sha256 of its files.json."""
    package, out = Path(package), Path(out)
    if not (package / "Manifest.json").is_file():
        raise FileNotFoundError(f"{package} is not an .mlpackage (no Manifest.json); run `just stage 4`")
    if out.exists():
        shutil.rmtree(out)
    (out / TOKENIZER_DIR).mkdir(parents=True)
    _clone_tree(package, out / PACKAGE_NAME)
    for name in TOKENIZER_FILES:
        shutil.copyfile(Path(tokenizer) / name, out / TOKENIZER_DIR / name)
    return write_files_manifest(out)
