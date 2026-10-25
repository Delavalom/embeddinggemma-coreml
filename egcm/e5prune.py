"""Background pruning of Core ML's E5 cache while a stage evaluates (s4).

Measured 2026-10-24 (s2, CPU_ONLY FP32): Core ML writes a new bundle of about 1.1 GB (one copy of the function's
weights) every 40 to 140 s, even while one function stays loaded, so deleting bundles only when switching functions
(egcm.e5cache) still let the cache reach 13 GB. `BundlePruner` runs a thread that, every `interval` s, deletes this
process's large bundles (> `min_mb`) created since it started and older than `min_age_s`, except the newest large one.
A loaded model has already mapped its bundle, so deleting an older one does not disturb it (checked in s2 and s3).
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

from egcm.e5cache import bundles


def _mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


class BundlePruner:
    def __init__(self, interval: float = 30.0, min_age_s: float = 120.0, min_mb: float = 100.0):
        self.interval, self.min_age_s, self.min_mb = interval, min_age_s, min_mb
        self.before = bundles()
        self.deleted_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def prune_once(self) -> None:
        now = time.time()
        big = []
        for p in bundles() - self.before:
            try:
                mb = _mb(p)
                if mb > self.min_mb:
                    big.append((p.stat().st_mtime, p, mb))
            except FileNotFoundError:
                continue
        big.sort(reverse=True)
        for mtime, p, mb in big[1:]:
            if now - mtime > self.min_age_s:
                shutil.rmtree(p, ignore_errors=True)
                self.deleted_mb += mb

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.prune_once()

    def __enter__(self) -> BundlePruner:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.prune_once()
