"""Where a report came from, and whether it is still valid.

A report is stale when the code that produced it (`code_hash`) or the inputs it read
(`inputs_sha256`) differ from what they are now. See AGENTS.md, "Provenance and staleness".
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from egcm.paths import HF_REVISION, ROOT
from egcm.report import Report, read_report


class MissingInputError(RuntimeError):
    """An input the stage reads (an upstream report or the fixture manifest) does not exist."""


@dataclass(frozen=True)
class StageInputs:
    upstream: tuple[str, ...] = ()  # stages whose report.json this stage reads
    fixtures_manifest: bool = False  # reads fixtures/MANIFEST.sha256
    hf_revision: bool = False  # downloads HF_REVISION


STAGE_INPUTS: dict[str, StageInputs] = {
    "s0_reference": StageInputs(fixtures_manifest=True, hf_revision=True),
    "s1_rewrite": StageInputs(upstream=("s0_reference",)),
    "s2_coreml_fp32": StageInputs(upstream=("s0_reference", "s1_rewrite")),
    "s3_coreml_ane": StageInputs(upstream=("s0_reference", "s1_rewrite", "s2_coreml_fp32")),
    "s4_compress": StageInputs(upstream=("s0_reference", "s3_coreml_ane")),
    "s5_swift": StageInputs(upstream=("s0_reference", "s4_compress")),
    "s6_device": StageInputs(upstream=("s4_compress", "s5_swift")),
    "s7_publish": StageInputs(upstream=("s0_reference", "s1_rewrite", "s2_coreml_fp32", "s3_coreml_ane",
                                        "s4_compress", "s5_swift", "s6_device")),
}


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=check)


def git_sha(root: Path = ROOT) -> str:
    return _git(root, "rev-parse", "HEAD").stdout.decode().strip()


def watched_paths(stage: str) -> list[str]:
    """The paths whose content decides a stage's result (uv.lock fixes the installed package versions)."""
    return ["egcm", f"stages/{stage}", "gates.toml", "pyproject.toml", "uv.lock"]


def git_dirty(stage: str, root: Path = ROOT) -> bool:
    """True if any watched path differs from HEAD (modified, staged or untracked)."""
    return bool(_git(root, "status", "--porcelain", "--", *watched_paths(stage)).stdout.strip())


def code_hash(stage: str, root: Path = ROOT) -> str:
    """sha256 over `git rev-parse HEAD:<path>` for each watched path.

    If the tree is dirty, also hashes `git diff HEAD` and the contents of untracked files in those paths.
    """
    paths = watched_paths(stage)
    h = hashlib.sha256()
    for p in paths:
        r = _git(root, "rev-parse", f"HEAD:{p}", check=False)
        h.update(f"{p}={r.stdout.decode().strip() if r.returncode == 0 else 'absent'}\n".encode())
    if git_dirty(stage, root):
        h.update(_git(root, "diff", "HEAD", "--", *paths).stdout)
        untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", *paths).stdout
        for name in sorted(n for n in untracked.decode().split("\0") if n):
            h.update(b"untracked:" + name.encode() + b"\0" + (root / name).read_bytes())
    return h.hexdigest()


def upstream_reports(stage: str, root: Path = ROOT) -> list[Report]:
    """The reports this stage reads, in stage-name order. Raises MissingInputError if one is absent."""
    out = []
    for up in sorted(STAGE_INPUTS[stage].upstream):
        p = root / "artifacts" / up / "report.json"
        if not p.is_file():
            raise MissingInputError(f"{stage} needs {up}, but {p} does not exist; run `just stage {up[1]}` first")
        out.append(read_report(p))
    return out


def inputs_sha256(stage: str, root: Path = ROOT, hf_revision: str = HF_REVISION) -> str:
    """sha256 over the sorted (stage, code_hash, inputs_sha256) of each upstream report, plus
    fixtures/MANIFEST.sha256 and the HF revision for the stages that read them."""
    spec = STAGE_INPUTS[stage]
    h = hashlib.sha256()
    for r in upstream_reports(stage, root):
        h.update(f"upstream:{r.stage}:{r.code_hash}:{r.inputs_sha256}\n".encode())
    if spec.fixtures_manifest:
        m = root / "fixtures" / "MANIFEST.sha256"
        if not m.is_file():
            raise MissingInputError(f"{stage} needs {m}, which does not exist; run build_fixtures first")
        h.update(b"manifest:" + hashlib.sha256(m.read_bytes()).hexdigest().encode() + b"\n")
    if spec.hf_revision:
        h.update(f"hf_revision:{hf_revision}\n".encode())
    return h.hexdigest()


def _stage_number(stage: str) -> str:
    return stage.split("_")[0][1:]


def upstream_problem(stage: str, root: Path = ROOT) -> str | None:
    """Why this stage cannot rely on its upstream reports, or None if it can.

    An upstream report must exist, have status `passed`, and not be stale itself (checked recursively,
    so the message names the earliest stage that needs rerunning).
    """
    for up in sorted(STAGE_INPUTS[stage].upstream):
        n = _stage_number(up)
        p = root / "artifacts" / up / "report.json"
        if not p.is_file():
            return f"{up} has no report; run `just stage {n}`"
        try:
            r = read_report(p)
        except Exception:
            return f"{up} has an unreadable report; rerun `just stage {n}`"
        if r.status != "passed" or not r.passed:
            return f"{up} has status {r.status!r}; rerun `just stage {n}`"
        if why := stale_reason(r, root):
            return why
    return None


def stale_reason(report: Report, root: Path = ROOT) -> str | None:
    """Why a report is stale, or None if it is current.

    Stale means any of: an upstream report is missing, not passed or itself stale; the stage's code changed
    (`code_hash`); or its inputs changed (`inputs_sha256`).
    """
    stage = report.stage
    if stage not in STAGE_INPUTS:
        raise ValueError(f"unknown stage {stage!r}")
    if why := upstream_problem(stage, root):
        return why
    rerun = f"rerun `just stage {_stage_number(stage)}`"
    if report.code_hash != code_hash(stage, root):
        return f"{stage} is stale (code changed); {rerun}"
    try:
        if report.inputs_sha256 != inputs_sha256(stage, root):
            return f"{stage} is stale (inputs changed); {rerun}"
    except MissingInputError:
        return f"{stage} is stale (an input is missing); {rerun}"
    return None


def is_stale(report: Report, root: Path = ROOT) -> bool:
    """True if `stale_reason` finds anything: changed code or inputs, or an upstream report that is
    missing, not passed or stale (recursively)."""
    return stale_reason(report, root) is not None
