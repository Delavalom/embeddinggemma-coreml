"""Where a report came from, and whether it is still valid.

A report is stale when the code that produced it (`code_hash`) or the inputs it read
(`inputs_sha256`) differ from what they are now. See AGENTS.md, "Provenance and staleness".
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
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


# The paths whose code decides each stage's *outputs*. Infrastructure (egcm/stage.py, provenance.py, report.py,
# gates.py, scripts/) is deliberately absent: changing how reports are written does not change what a stage
# computes. The stage's own [table] in gates.toml is hashed as well (see gate_table); other tables are not.
_LOCK = ("pyproject.toml", "uv.lock")
_REWRITE = ("egcm/model", "egcm/config.py", "egcm/weights.py", "egcm/functions.py")
_BUILD = ("egcm/convert.py", "egcm/parts.py", "egcm/mfmerge.py", "egcm/evaluate.py")  # Core ML stages s2-s4
_COMPARE = ("egcm/buckets.py", "egcm/metrics.py", "egcm/fixtures.py", "egcm/paths.py")
STAGE_CODE: dict[str, tuple[str, ...]] = {
    "s0_reference": ("stages/s0_reference", "egcm/buckets.py", "egcm/fixtures.py", "egcm/paths.py", *_LOCK),
    # s1 also checks s0's recorded patch positions with s0's own pixels.py
    "s1_rewrite": ("stages/s1_rewrite", "stages/s0_reference/pixels.py", *_REWRITE, *_COMPARE, *_LOCK),
    "s2_coreml_fp32": ("stages/s2_coreml_fp32", *_BUILD, *_REWRITE, *_COMPARE, *_LOCK),
    "s3_coreml_ane": ("stages/s3_coreml_ane", *_BUILD, "egcm/placement.py", "egcm/probe.py", "egcm/e5cache.py",
                      "egcm/ranking.py", *_REWRITE, *_COMPARE, *_LOCK),
    "s4_compress": ("stages/s4_compress", "egcm/compress.py", *_BUILD, "egcm/placement.py", "egcm/e5cache.py",
                    "egcm/e5prune.py", "egcm/ranking.py", "egcm/archive.py", *_REWRITE, *_COMPARE, *_LOCK),
    # s5-s7 take s4's package and device archive through egcm/published.py and egcm/manifest.py
    "s5_swift": ("stages/s5_swift", "swift", "egcm/published.py", "egcm/manifest.py", *_COMPARE, *_LOCK),
    "s6_device": ("stages/s6_device", "swift", "Examples", "egcm/metrics.py", "egcm/paths.py", "egcm/published.py",
                  "egcm/manifest.py", "egcm/buckets.py", *_LOCK),
    "s7_publish": ("stages/s7_publish", "swift", "egcm/paths.py", "egcm/published.py", "egcm/manifest.py",
                   "egcm/buckets.py", *_LOCK),
}


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=check)


def git_sha(root: Path = ROOT) -> str:
    return _git(root, "rev-parse", "HEAD").stdout.decode().strip()


def watched_paths(stage: str) -> list[str]:
    """The code paths whose content decides a stage's outputs (uv.lock fixes the installed package versions)."""
    if stage not in STAGE_CODE:
        raise ValueError(f"unknown stage {stage!r}")
    return list(STAGE_CODE[stage])


def gate_table(stage: str, root: Path = ROOT, rev: str | None = None) -> str:
    """The stage's own table in gates.toml as canonical JSON (values only; comments and layout do not count).

    `rev` reads the committed file (e.g. "HEAD"); None reads the working tree. "absent" if there is no table."""
    if rev is None:
        p = root / "gates.toml"
        text = p.read_text() if p.is_file() else ""
    else:
        r = _git(root, "show", f"{rev}:gates.toml", check=False)
        text = r.stdout.decode() if r.returncode == 0 else ""
    table = tomllib.loads(text).get(stage) if text else None
    return "absent" if table is None else json.dumps(table, sort_keys=True)


def git_dirty(stage: str, root: Path = ROOT) -> bool:
    """True if any watched path differs from HEAD (modified, staged or untracked), or the stage's gate table does."""
    if _git(root, "status", "--porcelain", "--", *watched_paths(stage)).stdout.strip():
        return True
    return gate_table(stage, root) != gate_table(stage, root, rev="HEAD")


def code_hash(stage: str, root: Path = ROOT) -> str:
    """sha256 over `git rev-parse HEAD:<path>` for each watched path, plus the stage's gate table.

    If the code paths are dirty, also hashes `git diff HEAD` and the contents of untracked files in them. The gate
    table is hashed by value from the working tree, so it never needs the diff."""
    paths = watched_paths(stage)
    h = hashlib.sha256()
    for p in paths:
        r = _git(root, "rev-parse", f"HEAD:{p}", check=False)
        h.update(f"{p}={r.stdout.decode().strip() if r.returncode == 0 else 'absent'}\n".encode())
    h.update(f"gates:{gate_table(stage, root)}\n".encode())
    if _git(root, "status", "--porcelain", "--", *paths).stdout.strip():
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
