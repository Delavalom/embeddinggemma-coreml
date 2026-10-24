"""Fail when gates.toml changes without a new docs/decisions/ADR-*.md. Run as `python -m scripts.check_gates_adr`.

Pass the git range to check (CI does: the PR base or the pushed range). Without one it is `origin/main...HEAD` when
`CI` is set, else the last commit (`HEAD~1...HEAD`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from egcm.paths import ROOT


def violates(paths: list[str]) -> bool:
    """True if `gates.toml` (repo root) is among the changed paths and no ADR is."""
    return "gates.toml" in paths and not any(p.startswith("docs/decisions/ADR-") for p in paths)


def changed_paths(base: str, root: Path = ROOT) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", base], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def default_base() -> str:
    # GitHub Actions sets CI=true; there the branch is compared with main.
    return "origin/main...HEAD" if os.environ.get("CI") else "HEAD~1...HEAD"


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    """argv[0], if given, is the git range to check (CI passes `<base>...HEAD`)."""
    args = sys.argv[1:] if argv is None else argv
    base = args[0] if args else default_base()
    paths = changed_paths(base, root)
    if violates(paths):
        print(f"error: gates.toml changed in {base} without a new docs/decisions/ADR-*.md", file=sys.stderr)
        return 1
    print(f"ok: gates.toml unchanged or justified by an ADR ({base}, {len(paths)} changed paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
