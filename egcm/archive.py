"""The device download: one Apple Archive (LZFSE) holding the model package and the tokenizer (ADR-0007).

An iPhone downloads exactly this file and unpacks it with the AppleArchive framework into the published layout the
Swift `ModelStore` expects:

    EmbeddingGemma2.mlpackage/**            the package, as built
    tokenizer/tokenizer.json                what `TextEncoder` loads through swift-transformers' AutoTokenizer
    tokenizer/tokenizer_config.json

Layout contract (agreed with the app and s7): exactly these paths, LZFSE, no symlinks, no absolute or ".." paths,
deterministic order. `device_archive` stages exactly those files in a temporary folder next to `out_path` and runs
`aa archive -d <stage> -o <out> -a lzfse`, without ownership, times, flags, extended attributes or ACLs, and with
modes normalized (0644 files, 0755 folders). `aa` enumerates entries in sorted path order, so the same inputs give
the same bytes. s4 gates the size of this file (`max_download_mb`); s7 publishes it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

AA = "/usr/bin/aa"
PACKAGE_NAME = "EmbeddingGemma2.mlpackage"  # ModelStore.packageName; the archive uses it whatever the source is called
TOKENIZER_DIR = "tokenizer"  # ModelStore.tokenizerDir
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")  # Installer.requiredFiles
# everything but the path, type, mode and data: the archive must not depend on who built it, when, or where
EXCLUDED_FIELDS = "uid,gid,flg,mtm,btm,ctm,xat,acl,duz"


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)  # data only: no metadata or extended attributes from the source
    os.chmod(dst, 0o644)


def stage_files(package_dir: Path, tokenizer_dir: Path) -> list[tuple[Path, str]]:
    """(source, path in the archive) for every file the device needs, sorted by archive path."""
    package_dir, tokenizer_dir = Path(package_dir), Path(tokenizer_dir)
    if not (package_dir / "Manifest.json").is_file():
        raise FileNotFoundError(f"{package_dir} is not an .mlpackage (no Manifest.json)")
    if links := [p for p in package_dir.rglob("*") if p.is_symlink()]:
        raise ValueError(f"symlinks are not allowed in the device archive: {links[:3]}")
    files = [(p, f"{PACKAGE_NAME}/{p.relative_to(package_dir).as_posix()}")
             for p in package_dir.rglob("*") if p.is_file()]
    for name in TOKENIZER_FILES:
        if not (tokenizer_dir / name).is_file():  # a symlink source (the HF cache) is fine: the data is copied
            raise FileNotFoundError(f"{tokenizer_dir / name} is missing")
        files.append((tokenizer_dir / name, f"{TOKENIZER_DIR}/{name}"))
    return sorted(files, key=lambda f: f[1])


def device_archive(package_dir: Path, tokenizer_dir: Path, out_path: Path) -> Path:
    """Write the device download (`out_path`, conventionally `.aar`) for `package_dir` and the tokenizer files in
    `tokenizer_dir`; returns `out_path`. Replaces an existing file only once the new one is complete."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_path.parent, prefix=".device_archive_") as tmp:
        root = Path(tmp) / "root"
        for src, rel in stage_files(package_dir, tokenizer_dir):
            _copy(src, root / rel)
        for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
            os.chmod(d, 0o755)
        partial = Path(tmp) / "out.aar"
        subprocess.run([AA, "archive", "-d", str(root), "-o", str(partial), "-a", "lzfse",
                        "-exclude-field", EXCLUDED_FIELDS], check=True, capture_output=True, text=True)
        os.replace(partial, out_path)
    return out_path
