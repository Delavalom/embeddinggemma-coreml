"""No absolute local paths in anything published or committed: stage reports (docs/results/), docs/RESULTS.md and
the Hugging Face upload. Paths under the repo become repo-relative and the home folder becomes "~"."""

from __future__ import annotations

import json
import re
from pathlib import Path

from egcm.paths import ROOT

# A home folder (macOS or Linux), the way it appears in a report or a traceback.
HOME_PATH = re.compile(r"(?:/Users|/home)/[^/\s\"'`]+")


def relativize_text(text: str, root: Path = ROOT, home: Path | None = None) -> str:
    """`root/...` -> `...`, `root` -> `.`, then `home/...` -> `~/...` (text-level, so tracebacks are covered too)."""
    root_s, home_s = str(Path(root)), str(home or Path.home())
    end = r"(?=[\"'`\s]|$)"  # the whole path, not a prefix of a longer name
    text = re.sub(re.escape(root_s) + end, ".", text.replace(root_s + "/", ""))
    return re.sub(re.escape(home_s) + end, "~", text.replace(home_s + "/", "~/"))


def relativize(obj, root: Path = ROOT, home: Path | None = None):
    """The same for every string in a JSON-like value."""
    return json.loads(relativize_text(json.dumps(obj), root, home))


def home_paths(text: str) -> list[str]:
    """Every absolute home path in `text` (empty if it is safe to publish)."""
    return sorted(set(HOME_PATH.findall(text)))


def files_with_home_paths(paths) -> dict[str, list[str]]:
    """{file: home paths found} over the text files among `paths` (.md, .json, .txt, .swift)."""
    out = {}
    for p in map(Path, paths):
        if p.is_file() and p.suffix in (".md", ".json", ".txt", ".swift"):
            if found := home_paths(p.read_text(errors="replace")):
                out[str(p)] = found
    return out
