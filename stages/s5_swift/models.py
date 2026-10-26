"""Which package s5 runs for which image budget (ADR-0010).

s4 ships one package per budget. egcm-parity and stages.s5_swift.runtime take them as `--model <budget>=<pkg>`, once
per budget; a bare `--model <pkg>` still means one 14-function package for every budget (`--measure`, the contract
model). Image vectors of a budget come from that budget's package. Text vectors come from `TEXT_BUDGET`'s package:
both packages hold the same text functions (s4 checks every function of each against the 14-function package,
CPU_ONLY, bitwise), so one of them is enough, and using the same one everywhere keeps the comparison consistent.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from egcm.buckets import BUDGETS

TEXT_BUDGET = 70  # the package Python Core ML takes text vectors from
_BUDGET_ARG = re.compile(r"(\d+)=(.+)")


def parse_models(values: list[str]) -> dict[int, Path]:
    """`--model` values -> budget -> package. One bare path serves every budget."""
    if len(values) == 1 and not _BUDGET_ARG.fullmatch(values[0]):
        return {b: Path(values[0]) for b in BUDGETS}
    out: dict[int, Path] = {}
    for v in values:
        m = _BUDGET_ARG.fullmatch(v)
        if not m or int(m[1]) not in BUDGETS or int(m[1]) in out:
            raise ValueError(f"--model {v!r}: expected <budget>=<package> once per budget {BUDGETS}, or one package")
        out[int(m[1])] = Path(m[2])
    if sorted(out) != sorted(BUDGETS):
        raise ValueError(f"--model covers budgets {sorted(out)}, expected {list(BUDGETS)}")
    return out


def model_args(model: Path | Mapping[int, Path]) -> list[str]:
    """`--model` arguments: one bare path for a single package, else `<budget>=<pkg>` per budget."""
    if isinstance(model, Mapping):
        return [x for b, p in sorted(model.items()) for x in ("--model", f"{b}={p}")]
    return ["--model", str(model)]


def budget_of(function: str) -> int:
    """The budget whose package runs `function`: an image function's own, `TEXT_BUDGET` for text."""
    if function.startswith("embed_image_"):
        return int(function.split("_")[2])
    return TEXT_BUDGET
