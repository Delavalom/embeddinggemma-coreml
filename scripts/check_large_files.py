"""Fail on any tracked file over 5 MB (fixture images: over 2 MB). Run as `python -m scripts.check_large_files`.

Weights and .mlpackage files never belong in git (.gitignore keeps them out; this catches what slips through).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from egcm.paths import ROOT

MB = 1024 * 1024
LIMIT = 5 * MB
IMAGE_PREFIX = "fixtures/images/"
IMAGE_LIMIT = 2 * MB


def limit_for(path: str) -> int:
    return IMAGE_LIMIT if path.startswith(IMAGE_PREFIX) else LIMIT


def violations(sizes: dict[str, int]) -> list[str]:
    """Tracked paths over their limit, as messages, sorted by path."""
    return [f"{p}: {n / MB:.1f} MB > {limit_for(p) // MB} MB" for p, n in sorted(sizes.items()) if n > limit_for(p)]


def tracked_sizes(root: Path = ROOT) -> dict[str, int]:
    """Size of every tracked file as it is in the working tree (files deleted but not yet committed are skipped)."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    paths = [p for p in out.decode().split("\0") if p]
    return {p: (root / p).stat().st_size for p in paths if (root / p).is_file()}


def main() -> int:
    sizes = tracked_sizes()
    bad = violations(sizes)
    for line in bad:
        print(f"error: {line}", file=sys.stderr)
    if not bad:
        print(f"ok: {len(sizes)} tracked files within limits (5 MB; fixture images 2 MB)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
