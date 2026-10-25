"""s4 packaging helpers: the package zip (a recorded size, not the download), checksums, cleanup. The device
download is egcm.archive; the variant metrics are egcm.ranking (shared with s3; ADR-0006, ADR-0007)."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path


def zip_package(pkg: Path, dest: Path) -> Path:
    """`dest` (.zip, deflated) holding `pkg` as its top-level directory, the form the app downloads."""
    dest.unlink(missing_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for p in sorted(pkg.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(pkg.parent).as_posix())
    return dest


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checksums(pkg: Path, download: Path) -> dict:
    """sha256 of the device download (`download`, the Apple Archive) and of every file of the package."""
    files = {p.relative_to(pkg.parent).as_posix(): sha256(p) for p in sorted(pkg.rglob("*")) if p.is_file()}
    return {"archive": {download.name: sha256(download), "bytes": download.stat().st_size}, "files": files}


def remove(*paths: Path) -> None:
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
