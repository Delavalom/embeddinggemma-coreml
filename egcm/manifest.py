"""`manifest.json`: the device download's manifest at the published repo root (s7), and unpacking the archive
(s5 and s6 test exactly the bytes s4 gated).

Shared contract (the Swift `ArchiveManifest`, and Roll): a JSON object, unknown keys ignored,

    {"format": 1,
     "archive": "EmbeddingGemma2.aar",      a single path component next to manifest.json
     "size": <archive bytes>, "sha256": "<64 lowercase hex of the archive>",
     "unpacked_size": <bytes after extraction, optional>,
     "revision": "<label, optional, e.g. v1.0.0>",
     "functions": [optional; if present it contains every REQUIRED_FUNCTIONS name]}

The archive (egcm.archive.device_archive) holds exactly EmbeddingGemma2.mlpackage/** and
tokenizer/{tokenizer.json,tokenizer_config.json}: LZFSE, no symlinks, no absolute or ".." paths.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from egcm.buckets import BUCKETS
from egcm.published import PACKAGE_NAME, TOKENIZER_DIR, TOKENIZER_FILES, Shipped, sha256_file

MANIFEST_JSON = "manifest.json"
ARCHIVE_NAME = "EmbeddingGemma2.aar"
FORMAT = 1
AA = "/usr/bin/aa"
# The functions every consumer needs (the Swift package and Roll check them); the package's own names must match.
REQUIRED_FUNCTIONS = tuple(f"embed_text_{n}" for n in (16, 32, 64, 128)) + tuple(b.function_name for b in BUCKETS)


def check_functions(functions) -> list[str]:
    """The package's function names, sorted; raises unless they include every REQUIRED_FUNCTIONS name."""
    names = sorted(set(functions))
    if missing := [f for f in REQUIRED_FUNCTIONS if f not in names]:
        raise ValueError(f"the package lacks the functions consumers need: {missing} (it has {names})")
    return names


def check_layout(root: Path, label: str = "archive") -> int:
    """The unpacked layout at `root`: only EmbeddingGemma2.mlpackage/ (with Manifest.json) and tokenizer/ (exactly
    the two files), no symlinks. Returns the bytes of its regular files."""
    root = Path(root)
    top = sorted(p.name for p in root.iterdir())
    if top != sorted([PACKAGE_NAME, TOKENIZER_DIR]):
        raise ValueError(f"{label}: the root holds {top}, expected {PACKAGE_NAME} and {TOKENIZER_DIR}/")
    if sorted(p.name for p in (root / TOKENIZER_DIR).iterdir()) != sorted(TOKENIZER_FILES):
        raise ValueError(f"{label}: {TOKENIZER_DIR}/ must hold exactly {sorted(TOKENIZER_FILES)}")
    if not (root / PACKAGE_NAME / "Manifest.json").is_file():
        raise ValueError(f"{label}: {PACKAGE_NAME}/Manifest.json is missing")
    everything = list(root.rglob("*"))
    if links := [str(p.relative_to(root)) for p in everything if p.is_symlink()]:
        raise ValueError(f"{label}: symlinks are not allowed: {links[:3]}")
    return sum(p.stat().st_size for p in everything if p.is_file())


def unpack(archive: Path, dest: Path, sha256: str | None = None) -> Path:
    """Extracts `archive` to `dest` (replacing it) after checking its sha256 (if given), then checks the layout."""
    archive, dest = Path(archive), Path(dest)
    if sha256 is not None and sha256_file(archive) != sha256:
        raise ValueError(f"{archive} does not match its sha256 {sha256[:12]}...; rerun `just stage 4`")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    subprocess.run([AA, "extract", "-i", str(archive), "-d", str(partial)], check=True, capture_output=True)
    check_layout(partial, archive.name)
    shutil.rmtree(dest, ignore_errors=True)
    partial.rename(dest)
    return dest


def write(s4: Shipped, out: Path, revision: str | None = None) -> str:
    """Copies s4's archive (the gated bytes, sha256-checked) to out/EmbeddingGemma2.aar and writes out/manifest.json;
    returns the sha256 of manifest.json (what `defaultManifestSHA256` pins). The archive is unpacked once into a
    temporary folder to check the layout and measure `unpacked_size`."""
    out = Path(out)
    names = check_functions(s4.functions)
    out.mkdir(parents=True, exist_ok=True)
    aar = out / ARCHIVE_NAME
    if subprocess.run(["cp", "-c", str(s4.archive), str(aar)], capture_output=True).returncode != 0:
        shutil.copyfile(s4.archive, aar)
    sha = sha256_file(aar)
    if sha != s4.archive_sha256:
        raise ValueError(f"{s4.archive} does not match s4's checksums.json; rerun `just stage 4`")
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".unpack-") as tmp:
        unpacked = check_layout(unpack(aar, Path(tmp) / "x"), aar.name)
    manifest = {"format": FORMAT, "archive": aar.name, "size": aar.stat().st_size, "sha256": sha,
                "unpacked_size": unpacked, "functions": names}
    if revision:
        manifest["revision"] = revision
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (out / MANIFEST_JSON).write_bytes(data)
    return hashlib.sha256(data).hexdigest()
