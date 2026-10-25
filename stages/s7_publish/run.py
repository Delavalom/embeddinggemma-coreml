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
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from egcm import published
from egcm.paths import ROOT, STAGES
from egcm.provenance import stale_reason
from egcm.report import Report, read_report
from egcm.stage import run_stage

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
        if not (s4.package / "Manifest.json").is_file():
            out.append(f"{s4.package} (s4 recipe {s4.recipe}) is missing; rerun `just stage 4`")
    return out


def render_default_model(repo: str, revision: str, files_sha256: str) -> str:
    return f"""// The published Core ML model. Written by stages/s7_publish (Task 22): the Hugging Face repo, the commit
// SHA of the upload and the sha256 of its files.json.
public extension EmbeddingGemma {{
  static let defaultRepo = "{repo}"
  static let defaultRevision = "{revision}"
  /// sha256 of `files.json` at `defaultRevision`. When set, loading the default model verifies `files.json` against
  /// it (on top of the pinned commit), so a compromised or rewritten Hub file is rejected.
  static let defaultFilesManifestSHA256: String? = "{files_sha256}"
}}
"""


def model_card(repo: str, results_md: str, s4: published.Shipped | None = None) -> str:
    weights = ""
    if s4 is not None:
        mb = "" if s4.download_mb is None else f", {s4.download_mb:.0f} MB zipped"
        weights = f"\nWeights: s4 recipe `{s4.recipe}`{mb}. See the s4 table below.\n"
    return f"""---
license: apache-2.0
base_model: google/embeddinggemma-2
library_name: coreml
tags: [coreml, embeddings, image-text, ios, neural-engine]
---

# EmbeddingGemma 2 for Core ML

`google/embeddinggemma-2` (text and image embeddings in one 768-d space) converted to a multifunction Core ML
model for the iPhone Neural Engine, with parity checked against the original at every stage.
{weights}
Use it from Swift with the `EmbeddingGemma` package (`EmbeddingGemma.load(.huggingFace(repo: "{repo}", revision:
...))`). Every file is listed with its size and SHA-256 in `files.json` and verified on download.

License: Apache-2.0, same as the base model. See NOTICE.

{results_md}
"""


def build_upload(root: Path, repo: str, out: Path) -> str:
    """The upload folder: the published layout plus README.md (model card) and NOTICE. Returns files.json's sha256.
    files.json covers the model and tokenizer only (what ModelStore downloads), so it is written first."""
    s4 = published.shipped(root)
    files_sha = published.assemble(s4.package, published.tokenizer_source(root), out)
    (out / "README.md").write_text(model_card(repo, (root / RESULTS_MD).read_text(), s4))
    shutil.copyfile(root / "NOTICE", out / "NOTICE")
    return files_sha


def publish(root: Path, repo: str, tag: str, pub: Publisher) -> dict:
    """The irreversible part, in order. Returns what to record in the report's details."""
    upload = root / "artifacts" / STAGE / "upload"
    files_sha = build_upload(root, repo, upload)
    log(f"creating https://huggingface.co/{repo} (public) and uploading {upload}")
    pub.hf.create_repo(repo_id=repo, repo_type="model", private=False, exist_ok=True)
    info = pub.hf.upload_folder(repo_id=repo, repo_type="model", folder_path=str(upload),
                                commit_message=f"EmbeddingGemma 2 Core ML {tag}")
    revision = getattr(info, "oid", None)
    if not revision or not _SHA.fullmatch(revision):
        raise RuntimeError(f"the upload returned no commit SHA (got {revision!r})")
    (root / DEFAULT_MODEL_SWIFT).write_text(render_default_model(repo, revision, files_sha))
    pub.check(["git", "add", str(DEFAULT_MODEL_SWIFT)])
    pub.check(["git", "commit", "-m", f"chore(release): pin the default model to {repo}@{revision[:12]}"])
    pub.check(["git", "tag", "-a", tag, "-m", f"EmbeddingGemma {tag} (model {repo}@{revision})"])
    log("making the GitHub repo public, pushing, and creating the release")
    pub.check(["gh", "repo", "edit", "--visibility", "public", "--accept-visibility-change-consequences"])
    pub.check(["git", "push", "origin", "HEAD"])
    pub.check(["git", "push", "origin", tag])
    pub.check(["gh", "release", "create", tag, "--title", f"EmbeddingGemma {tag}",
               "--notes-file", str(RESULTS_MD)])
    return {"hf_repo": repo, "revision": revision, "files_json_sha256": files_sha, "tag": tag}


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
