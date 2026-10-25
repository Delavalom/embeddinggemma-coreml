# group: reference
"""s7_publish: upload the model to Hugging Face, pin it in the Swift package, tag and release v1.0.0.

Guarded. It refuses to do anything unless
  1. `--confirm` is given (only after the founder approves the final numbers), and
  2. HF_REPO_TARGET (`owner/name`) is set, and
  3. s0-s6 all have status `passed` and are not stale, and
  4. the git tree is clean, the tag does not exist yet, and docs/RESULTS.md exists.

    HF_REPO_TARGET=owner/name just publish --confirm

Gate (gates.toml [s7_publish]): confirmed. See stages/s7_publish/README.md. It runs in the `reference` group
because that is the locked environment with `huggingface_hub`; the token comes from the logged-in `hf` CLI or
HF_TOKEN, never from this script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from egcm import manifest, published
from egcm.paths import ROOT, STAGES
from egcm.provenance import stale_reason
from egcm.report import Report, read_report
from egcm.stage import run_stage
from stages.s7_publish import card

STAGE = "s7_publish"
UPSTREAM = STAGES[:7]  # s0_reference .. s6_device
DEFAULT_TAG = "v1.0.0"
DEFAULT_MODEL_SWIFT = Path("swift/Sources/EmbeddingGemma/DefaultModel.swift")
RESULTS_MD = Path("docs/RESULTS.md")
_REPO = re.compile(r"[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+")
_SHA = re.compile(r"[0-9a-f]{40}")


def log(msg: str) -> None:
    print(f"[s7] {msg}", flush=True)


def upstream_problems(root: Path = ROOT) -> list[str]:
    """Why s0-s6 do not allow a release; empty if every report is passed and current."""
    out = []
    for stage in UPSTREAM:
        p = root / "artifacts" / stage / "report.json"
        if not p.is_file():
            out.append(f"{stage}: no report")
            continue
        try:
            r = read_report(p)
        except Exception as e:
            out.append(f"{stage}: unreadable report ({e})")
            continue
        if r.status != "passed" or not r.passed:
            out.append(f"{stage}: status {r.status!r}")
        elif why := stale_reason(r, root):
            out.append(f"{stage}: {why}")
    return out


Runner = Callable[[list[str]], subprocess.CompletedProcess]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


class Publisher:
    """Every side effect of s7, so tests replace it. `hf` is a `huggingface_hub.HfApi`, imported lazily."""

    def __init__(self, run: Runner = _run, hf=None):
        self.run, self._hf = run, hf

    @property
    def hf(self):
        if self._hf is None:
            from huggingface_hub import HfApi  # only in the reference group

            self._hf = HfApi()
        return self._hf

    def check(self, cmd: list[str]) -> str:
        r = self.run(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"`{' '.join(cmd)}` failed ({r.returncode}): {(r.stderr or r.stdout).strip()}")
        return r.stdout


def preflight(root: Path, tag: str, pub: Publisher) -> list[str]:
    """Local checks that need no network. Empty if publishing may start."""
    out = []
    if pub.run(["git", "status", "--porcelain"]).stdout.strip():
        out.append("the git tree is dirty; commit or stash first (the tag must point at what was tested)")
    if pub.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"]).returncode == 0:
        out.append(f"tag {tag} already exists")
    if not (root / RESULTS_MD).is_file():
        out.append(f"{RESULTS_MD} is missing; run `python -m scripts.gen_results`")
    try:  # the package s4's report says it shipped, not a recipe name hard-coded here
        s4 = published.shipped(root)
    except Exception as e:  # no report, unreadable, or no winner
        out.append(f"s4_compress: {e}")
    else:
        for p in (s4.package / "Manifest.json", s4.archive):
            if not p.is_file():
                out.append(f"{p} (s4 recipe {s4.recipe}) is missing; rerun `just stage 4`")
        try:
            manifest.check_functions(s4.functions)
        except ValueError as e:
            out.append(f"s4_compress: {e}")
    return out


def render_default_model(repo: str, revision: str, manifest_sha256: str) -> str:
    return f"""// The published Core ML model. Written by stages/s7_publish (Task 22): the Hugging Face repo, the commit
// SHA of the upload and the sha256 of its manifest.json.
public extension EmbeddingGemma {{
  static let defaultRepo = "{repo}"
  static let defaultRevision = "{revision}"
  /// sha256 of `manifest.json` at `defaultRevision`. When set, loading the default model verifies `manifest.json`
  /// against it (on top of the pinned commit), so a compromised or rewritten Hub file is rejected.
  static let defaultManifestSHA256: String? = "{manifest_sha256}"
}}
"""


def model_card(repo: str, results_md: str, values: dict[str, str] | None = None, root: Path = ROOT) -> str:
    """The Hugging Face README: stages/s7_publish/model_card.md filled from the reports under `root` (card.py)."""
    return card.render(repo, results_md, card.values(root) if values is None else values)


# Shipped next to the model (Apache-2.0 section 4(a) and 4(d): the license text and the NOTICE).
UPLOAD_EXTRAS = ("LICENSE", "NOTICE")
RELEASE_NOTES_DIR = Path("docs/release-notes")


def build_upload(root: Path, repo: str, tag: str, out: Path) -> dict:
    """The upload folder: the plain published layout (files.json covers the package and tokenizer only, so it is
    written first), s4's device archive with manifest.json (what apps download), README.md (model card), LICENSE
    and NOTICE. Returns the sha256 of files.json and of manifest.json."""
    s4 = published.shipped(root)
    files_sha = published.assemble(s4.package, published.tokenizer_source(root), out)
    manifest_sha = manifest.write(s4, out, revision=tag)  # s4's gated archive, as is
    vals = card.values(root, json.loads((out / manifest.MANIFEST_JSON).read_text()))
    (out / "README.md").write_text(model_card(repo, (root / RESULTS_MD).read_text(), vals, root))
    copy_extras(root, out)
    return {"files_json_sha256": files_sha, "manifest_json_sha256": manifest_sha}

def copy_extras(root: Path, out: Path) -> None:
    for name in UPLOAD_EXTRAS:
        shutil.copyfile(root / name, out / name)


def release_notes(root: Path, tag: str, out: Path) -> Path:
    """The GitHub release notes: docs/release-notes/<tag>.md (hand-written) followed by docs/RESULTS.md (generated),
    written to `out`. Without a hand-written draft, just RESULTS.md."""
    draft = root / RELEASE_NOTES_DIR / f"{tag}.md"
    results = (root / RESULTS_MD).read_text()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text((draft.read_text().rstrip() + "\n\n" + results) if draft.is_file() else results)
    return out


def publish(root: Path, repo: str, tag: str, pub: Publisher) -> dict:
    """The irreversible part, in order. Returns what to record in the report's details."""
    upload = root / "artifacts" / STAGE / "upload"
    shas = build_upload(root, repo, tag, upload)
    log(f"creating https://huggingface.co/{repo} (public) and uploading {upload}")
    pub.hf.create_repo(repo_id=repo, repo_type="model", private=False, exist_ok=True)
    info = pub.hf.upload_folder(repo_id=repo, repo_type="model", folder_path=str(upload),
                                commit_message=f"EmbeddingGemma 2 Core ML {tag}")
    revision = getattr(info, "oid", None)
    if not revision or not _SHA.fullmatch(revision):
        raise RuntimeError(f"the upload returned no commit SHA (got {revision!r})")
    log(f"uploaded commit {revision}, sha256(manifest.json) {shas['manifest_json_sha256']}")
    (root / DEFAULT_MODEL_SWIFT).write_text(render_default_model(repo, revision, shas["manifest_json_sha256"]))
    pub.check(["git", "add", str(DEFAULT_MODEL_SWIFT)])
    pub.check(["git", "commit", "-m", f"chore(release): pin the default model to {repo}@{revision[:12]}"])
    pub.check(["git", "tag", "-a", tag, "-m", f"EmbeddingGemma {tag} (model {repo}@{revision})"])
    log("making the GitHub repo public, pushing, and creating the release")
    pub.check(["gh", "repo", "edit", "--visibility", "public", "--accept-visibility-change-consequences"])
    pub.check(["git", "push", "origin", "HEAD"])
    pub.check(["git", "push", "origin", tag])
    notes = release_notes(root, tag, root / "artifacts" / STAGE / "release_notes.md")
    pub.check(["gh", "release", "create", tag, "--title", f"EmbeddingGemma {tag}", "--notes-file", str(notes)])
    log(f"published https://huggingface.co/{repo}/resolve/{revision}/{manifest.MANIFEST_JSON} "
        f"(sha256 {shas['manifest_json_sha256']})")
    return {"hf_repo": repo, "revision": revision, **shas, "tag": tag}


def make_fill(root: Path, repo: str, tag: str, pub: Publisher):
    def fill(report: Report) -> None:
        # run_stage already refuses on a bad upstream report; checked again here so nothing below can run without it.
        if problems := upstream_problems(root):
            raise RuntimeError("s0-s6 are not all passed and current: " + "; ".join(problems))
        if problems := preflight(root, tag, pub):
            raise RuntimeError("; ".join(problems))
        report.metrics["confirmed"] = True
        report.details["s4_recipe"] = published.shipped(root).recipe
        report.details.update(publish(root, repo, tag, pub))
    return fill


def main(argv: list[str] | None = None, *, root: Path = ROOT, env=os.environ, pub: Publisher | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--confirm", action="store_true", help="really publish (only after the founder approves)")
    p.add_argument("--tag", default=DEFAULT_TAG)
    args = p.parse_args(argv)
    problems = upstream_problems(root)
    if not args.confirm:
        log("refusing: --confirm not given. Nothing was done and no report was written.")
        for line in problems or ["s0-s6 are passed and current"]:
            log(f"  {line}")
        return 2
    repo = env.get("HF_REPO_TARGET", "")
    if not _REPO.fullmatch(repo):
        log(f"refusing: set HF_REPO_TARGET to the Hugging Face repo (owner/name), got {repo!r}")
        return 2
    run_stage(STAGE, make_fill(root, repo, args.tag, pub or Publisher()), root=root)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
