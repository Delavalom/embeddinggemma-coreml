"""Keep Core ML's E5 runtime cache from filling the disk while a stage loads many functions (s3, s4).

Every function a Python process loads leaves a bundle in `~/Library/Caches/<python>/com.apple.e5rt.e5bundlecache`
about the size of the function's weights (measured 2026-10-23: 1.1 GB per FP32 function, 10 functions = 11 GB, never
reclaimed by the system while the disk filled). `PurgingPackage` deletes the bundles that appeared since it was
created whenever it switches to another function (the previous model is released first), and on `close()`. Bundles
that existed before, from other processes, are never touched.
"""

from __future__ import annotations

import gc
import shutil
from pathlib import Path

import coremltools as ct

from egcm.evaluate import Package

CACHE_ROOTS = tuple(Path.home() / "Library" / "Caches" / name / "com.apple.e5rt.e5bundlecache"
                    for name in ("python3", "python", "python3.12"))


def bundles() -> set[Path]:
    """Every cache entry directory (<cache>/<os build>/<hash>)."""
    return {p for root in CACHE_ROOTS if root.is_dir() for build in root.iterdir() if build.is_dir()
            for p in build.iterdir()}


class PurgingPackage(Package):
    def __init__(self, compiled: Path, compute_units: ct.ComputeUnit):
        super().__init__(compiled, compute_units)
        self._before = bundles()
        self.purged_mb = 0.0

    def purge(self) -> None:
        for p in bundles() - self._before:
            self.purged_mb += sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
            shutil.rmtree(p, ignore_errors=True)

    def function(self, name: str):
        if name != self._name and self._model is not None:
            self._model, self._name = None, None
            gc.collect()
            self.purge()
        return super().function(name)

    def close(self) -> None:
        self._model, self._name = None, None
        gc.collect()
        self.purge()
