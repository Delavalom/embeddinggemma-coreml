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
}  # fmt: skip


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=check)


def git_sha(root: Path = ROOT) -> str:
    return _git(root, "rev-parse", "HEAD").stdout.decode().strip()


def watched_paths(stage: str) -> list[str]:
    """The paths whose content decides a stage's result."""
    return ["egcm", f"stages/{stage}", "gates.toml", "pyproject.toml"]


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
            raise MissingInputError(f"{stage} needs {up}, but {p} does not exist; run `just stage` for it first")
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


def is_stale(report: Report, root: Path = ROOT) -> bool:
    """True if the code or the inputs changed since the report was made (or the inputs can't be found)."""
    if report.stage not in STAGE_INPUTS:
        raise ValueError(f"unknown stage {report.stage!r}")
    if report.code_hash != code_hash(report.stage, root):
        return True
    try:
        return report.inputs_sha256 != inputs_sha256(report.stage, root)
    except MissingInputError:
        return True
