"""Fixture files: the texts and images every stage compares on, and the manifest that pins them.

`fixtures/` is written only by `python -m stages.s0_reference.build_fixtures`. `MANIFEST.sha256` lists every
file under `fixtures/` except itself, as sorted `sha256  relative/path` lines.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from egcm.paths import FIXTURES

MANIFEST = "MANIFEST.sha256"


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def load_texts(root: Path = FIXTURES) -> list[dict]:
    """Rows of texts.jsonl: {"id", "text", "kind"} with kind "query" or "document", in file order."""
    return _read_jsonl(Path(root) / "texts.jsonl")


def load_images(root: Path = FIXTURES) -> list[dict]:
    """Rows of images.jsonl, in file order, each with an added absolute "path". Open one with `open_image`."""
    root = Path(root)
    return [{**r, "path": str(root / r["file"])} for r in _read_jsonl(root / "images.jsonl")]


def open_image(row: dict) -> Image.Image:
    """The fixture image as an RGB PIL image (fixtures are stored upright, without EXIF orientation)."""
    with Image.open(row["path"]) as im:
        return im.convert("RGB")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_files(root: Path = FIXTURES) -> list[str]:
    """Every file under root except the manifest and dotfiles, as sorted POSIX paths relative to root."""
    root = Path(root)
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != MANIFEST and not p.name.startswith(".")
    )


def render_manifest(root: Path = FIXTURES) -> str:
    root = Path(root)
    return "".join(f"{sha256_file(root / rel)}  {rel}\n" for rel in manifest_files(root))


def write_manifest(root: Path = FIXTURES) -> Path:
    p = Path(root) / MANIFEST
    p.write_text(render_manifest(root))
    return p


def verify_manifest(root: Path = FIXTURES) -> list[str]:
    """Mismatches between MANIFEST.sha256 and the files on disk. Empty means the fixtures are intact."""
    root = Path(root)
    m = root / MANIFEST
    if not m.is_file():
        return [f"{m} does not exist; run `python -m stages.s0_reference.build_fixtures`"]
    listed: dict[str, str] = {}
    for line in m.read_text().splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            listed[rel] = digest
    on_disk = set(manifest_files(root))
    problems = [f"missing: {rel}" for rel in sorted(set(listed) - on_disk)]
    problems += [f"not in manifest: {rel}" for rel in sorted(on_disk - set(listed))]
    problems += [
        f"hash mismatch: {rel}" for rel in sorted(set(listed) & on_disk) if sha256_file(root / rel) != listed[rel]
    ]
    return problems
