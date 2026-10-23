"""Report schema and file I/O. Gate logic is in egcm.gates; provenance is in egcm.provenance."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Status = Literal["running", "passed", "failed", "error"]
Metric = float | int | str | bool | None

# JSON has no NaN/Infinity, so non-finite floats are written as these strings and restored on read.
_NONFINITE = {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": float("-inf")}


class Report(BaseModel):
    """One stage run. `passed` is true only when `status == "passed"`."""

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    stage: str
    status: Status = "running"
    passed: bool = False
    failures: list[str] = Field(default_factory=list)
    metrics: dict[str, Metric] = Field(default_factory=dict)
    gate_values: dict[str, float | int | bool] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    # Provenance, stamped by egcm.stage.run_stage.
    git_sha: str = ""
    git_dirty: bool = False
    code_hash: str = ""
    inputs_sha256: str = ""
    started_at: str | None = None
    finished_at: str | None = None

    @field_validator("metrics", mode="after")
    @classmethod
    def _restore_nonfinite(cls, v: dict[str, Metric]) -> dict[str, Metric]:
        return {k: (_NONFINITE.get(x, x) if isinstance(x, str) else x) for k, x in v.items()}


def write_report(r: Report, directory: Path) -> Path:
    """Write report.json atomically (temp file + os.replace), so readers never see a partial file."""
    p = Path(directory) / "report.json"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(r.model_dump_json(indent=2) + "\n")
    os.replace(tmp, p)
    return p


def read_report(p: Path) -> Report:
    return Report.model_validate_json(Path(p).read_text())
