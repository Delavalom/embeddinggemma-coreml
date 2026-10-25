"""Empty Core ML's E5 bundle cache of what a process created, when that process exits (every stage, scripts).

Every Core ML model load leaves a compiled bundle in `~/Library/Caches/<executable>/com.apple.e5rt.e5bundlecache`,
about one copy of the model's weights, and the system does not reclaim it: s2 (CPU_ONLY, FP32) left 14 GB on
2026-10-25. `ExitPurge` snapshots the cache when entered and deletes every bundle that appeared since, when the block
exits: normally, on an exception, on KeyboardInterrupt, or on SIGTERM (turned into SystemExit while the block runs).
`egcm.stage.run_stage` wraps every stage in it. Bundles that existed before are never touched. A bundle that another
process created in the same window is deleted too; Core ML compiles it again on that process's next load.
Standard library only, so the reference environment (s0) can import it. SIGKILL cannot be caught: after `kill -9`,
delete the bundles by hand.
"""

from __future__ import annotations

import shutil
import signal
import sys
import threading
from pathlib import Path

# The cache folder is named after the executable (python3 for the uv venv), as in egcm.e5cache.
CACHE_ROOTS = (Path.home() / "Library" / "Caches" / Path(sys.executable).name / "com.apple.e5rt.e5bundlecache",)


def bundles(roots: tuple[Path, ...] | None = None) -> set[Path]:
    """Every cache entry directory (<cache>/<os build>/<hash>)."""
    roots = CACHE_ROOTS if roots is None else roots
    return {p for root in roots if root.is_dir() for build in root.iterdir() if build.is_dir()
            for p in build.iterdir()}


def _mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def purge_new(before: set[Path], roots: tuple[Path, ...] | None = None) -> float:
    """Delete the bundles not in `before`; returns the MB deleted."""
    mb = 0.0
    for p in bundles(roots) - before:
        mb += _mb(p)
        shutil.rmtree(p, ignore_errors=True)
    return mb


def _sigterm(signum, frame):
    raise SystemExit(128 + signum)


class ExitPurge:
    def __init__(self, roots: tuple[Path, ...] | None = None):
        self.roots, self.deleted_mb, self._prev = roots, 0.0, None

    def __enter__(self) -> ExitPurge:
        self.before = bundles(self.roots)
        if threading.current_thread() is threading.main_thread():
            self._prev = signal.signal(signal.SIGTERM, _sigterm)
        return self

    def __exit__(self, *exc) -> bool:
        if self._prev is not None:
            signal.signal(signal.SIGTERM, self._prev)
        self.deleted_mb = purge_new(self.before, self.roots)
        if self.deleted_mb:
            print(f"e5exit: deleted {self.deleted_mb:.0f} MB of Core ML E5 bundles this run created", flush=True)
        return False
