"""Load gates.toml and evaluate a report against the gates of its stage."""

from __future__ import annotations

import math
import numbers
import tomllib
from pathlib import Path

from egcm.paths import GATES
from egcm.report import Report


def _table(stage_key: str, path: Path) -> dict:
    t = tomllib.loads(Path(path).read_text()).get(stage_key)
    if not t:
        raise ValueError(f"gates.toml has no gates for stage {stage_key!r}")
    return t


def op_for(key: str, value: object) -> str:
    """Comparison direction from the key: bool -> ==, min_* -> >=, max_* -> <=. Anything else is an error."""
    if isinstance(value, bool):
        return "=="
    if not isinstance(value, int | float):
        raise ValueError(f"gate {key!r} must be a number or a bool, got {value!r}")
    if key.startswith("min_"):
        return ">="
    if key.startswith("max_"):
        return "<="
    raise ValueError(f"gate {key!r} must start with 'min_' or 'max_' (or be a bool)")


def ops_for(g: dict) -> dict[str, str]:
    return {k: op_for(k, v) for k, v in g.items()}


def gates(stage_key: str, path: Path = GATES) -> dict:
    """The gated values of a stage. The recorded-only `targets` subtable is not included."""
    g = {k: v for k, v in _table(stage_key, path).items() if k != "targets"}
    ops_for(g)  # validate every key now
    if not g:
        raise ValueError(f"gates.toml has no gates for stage {stage_key!r}")
    return g


def targets(stage_key: str, path: Path = GATES) -> dict:
    """Recorded-only numbers for a stage (the `targets` subtable). Empty if there are none."""
    return dict(_table(stage_key, path).get("targets", {}))


def _failure(key: str, op: str, gate: float | bool, r: Report) -> str | None:
    if key not in r.metrics or r.metrics[key] is None:
        return f"{key}: metric missing"
    metric = r.metrics[key]
    if op == "==":
        if not isinstance(metric, bool):
            return f"{key}: expected a bool, got {metric!r}"
        return None if metric == gate else f"{key}: {metric} not == {gate}"
    if isinstance(metric, bool) or not isinstance(metric, numbers.Real):
        return f"{key}: non-numeric metric {metric!r}"
    value = float(metric)
    if not math.isfinite(value):
        return f"{key}: non-finite metric {value}"
    ok = value >= gate if op == ">=" else value <= gate
    return None if ok else f"{key}: {value} not {op} {gate}"


def evaluate(r: Report, stage_key: str, path: Path = GATES) -> Report:
    """Set gate_values from gates.toml, check every metric against them, and set passed/failures/status.

    A missing, non-numeric or non-finite metric is a recorded failure, never an exception.
    Raises ValueError if the stage has no gates.
    """
    g = gates(stage_key, path)
    r.gate_values = g
    ops = ops_for(g)
    failures = [f for k, v in g.items() if (f := _failure(k, ops[k], v, r))]
    r.failures, r.passed = failures, not failures
    r.status = "passed" if r.passed else "failed"
    return r
