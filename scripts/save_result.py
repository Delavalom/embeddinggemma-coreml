"""Copy a stage report into docs/results/ for committing, with repo-relative paths and no home folder.

    python -m scripts.save_result s6_device            # docs/results/s6_<device model>.json (one per device)
    python -m scripts.save_result s6_device my_name    # docs/results/my_name.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from egcm.paths import ROOT
from egcm.redact import home_paths, relativize

RESULTS = Path("docs/results")


def save(stage: str, name: str | None = None, root: Path = ROOT) -> Path:
    data = relativize(json.loads((root / "artifacts" / stage / "report.json").read_text()), root)
    if name is None:
        devices = sorted((data.get("details") or {}).get("devices") or {})
        name = f"{stage.split('_')[0]}_{'+'.join(devices)}" if devices else stage
    text = json.dumps(data, indent=2) + "\n"
    if found := home_paths(text):
        raise ValueError(f"{stage}: the report still names {found}; refusing to write it to {RESULTS}")
    out = root / RESULTS / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def main(argv: list[str]) -> None:
    if not argv or len(argv) > 2:
        sys.exit(__doc__)
    print(save(*argv))


if __name__ == "__main__":
    main(sys.argv[1:])
