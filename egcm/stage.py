"""The one way a stage produces a report.

    def main():
        run_stage("s1_rewrite", fill)

    def fill(report):  # set report.metrics and report.details; do not touch status or passed
        report.metrics["min_cosine_text"] = ...
"""

from __future__ import annotations

import datetime as dt
import traceback
from collections.abc import Callable
from pathlib import Path

from egcm.gates import evaluate
from egcm.paths import ROOT, stage_dir
from egcm.provenance import code_hash, git_dirty, git_sha, inputs_sha256, upstream_reports
from egcm.report import Report, write_report


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _plain(v):
    """numpy scalars -> Python scalars, so metrics serialize."""
    return v.item() if hasattr(v, "item") and not isinstance(v, str | bytes) else v


def _stamp(r: Report, root: Path) -> None:
    for up in upstream_reports(r.stage, root):
        if up.status != "passed":
            raise RuntimeError(f"upstream stage {up.stage} has status {up.status!r}; it must pass first")
    r.git_sha = git_sha(root)
    r.git_dirty = git_dirty(r.stage, root)
    r.code_hash = code_hash(r.stage, root)
    r.inputs_sha256 = inputs_sha256(r.stage, root)


def run_stage(stage_key: str, fn: Callable[[Report], None], *, root: Path = ROOT) -> Report:
    """Run `fn(report)`, which fills in metrics and details, and write the evaluated report.

    1. Writes a `running` report first, so an older passing report can't survive a crash.
    2. Stamps provenance (git sha, dirty flag, code hash, inputs hash).
    3. Calls `fn`. An exception writes status `error` with the traceback in details.
    4. Evaluates against the gates of `stage_key` and writes the report atomically.
    5. Raises SystemExit(1) if the stage did not pass.
    """
    d = stage_dir(stage_key, root)
    r = Report(stage=stage_key, started_at=_now())
    write_report(r, d)
    try:
        _stamp(r, root)
        fn(r)
        r.metrics = {k: _plain(v) for k, v in r.metrics.items()}
        evaluate(r, stage_key, root / "gates.toml")
    except Exception as e:
        r.status, r.passed = "error", False
        r.failures = [f"{type(e).__name__}: {e}"]
        r.details["traceback"] = traceback.format_exc()
    r.finished_at = _now()
    try:
        write_report(r, d)
    except Exception as e:  # e.g. a detail that is not JSON-serializable
        r = Report(stage=stage_key, status="error", failures=[f"could not write report: {e!r}"],
                   started_at=r.started_at, finished_at=_now(), git_sha=r.git_sha)
        write_report(r, d)
    if not r.passed:
        raise SystemExit(1)
    return r
