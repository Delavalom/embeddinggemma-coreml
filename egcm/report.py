from __future__ import annotations
import subprocess, datetime as dt
from pathlib import Path
from pydantic import BaseModel, Field

class Report(BaseModel):
    stage: str
    git_sha: str
    inputs_sha256: str
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)
    gate_values: dict[str, float | int | bool] = Field(default_factory=dict)
    details: dict = Field(default_factory=dict)
    passed: bool = False
    failures: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
    finished_at: str | None = None

def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

def evaluate(r: Report, ops: dict[str, str]) -> Report:
    failures = []
    for key, gate in r.gate_values.items():
        if key not in r.metrics:
            failures.append(f"{key}: metric missing")
            continue
        val, op = r.metrics[key], ops.get(key, ">=")
        ok = {">=": val >= gate, "<=": val <= gate, "==": val == gate}[op]
        if not ok:
            failures.append(f"{key}: {val} not {op} {gate}")
    r.failures, r.passed = failures, not failures
    r.finished_at = dt.datetime.now(dt.UTC).isoformat()
    return r

def write_report(r: Report, directory: Path) -> Path:
    p = Path(directory) / "report.json"
    p.write_text(r.model_dump_json(indent=2))
    return p

def read_report(p: Path) -> Report:
    return Report.model_validate_json(Path(p).read_text())
