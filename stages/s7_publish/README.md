# s7_publish

Publishes v1.0.0: the model on Hugging Face, the Swift package pinned to it, the git tag, the GitHub release, and the
GitHub repo made public. **Run it only after the founder approves the final numbers.**

```
HF_REPO_TARGET=<owner>/<name> just publish --confirm
```

It runs in the `reference` group (the locked environment with `huggingface_hub`). The Hugging Face token comes
from the logged-in `hf` CLI or `HF_TOKEN`; `gh` must be logged in with rights to change the repo's visibility.

## Guards (in this order, nothing is changed until all pass)

1. No `--confirm`: prints why it refuses (and the s0–s6 status), exits 2, writes no report. A refusal never
   overwrites the report of a real publish.
2. `HF_REPO_TARGET` unset or not `owner/name`: exits 2, no report.
3. `run_stage` refuses (status `error`) unless s0–s6 are all `passed` and not stale; `fill` checks it again.
4. Preflight: clean git tree, the tag does not exist, `docs/RESULTS.md` exists (Task 21's `gen_results`), and s4's
   package is present.

Only then is `metrics.confirmed = true` set (the one gate, `[s7_publish].confirmed`).

## What it does

1. Builds `artifacts/s7_publish/upload/`: the published layout of `egcm.published` (s4's
   `EmbeddingGemma2.mlpackage/` uncompressed, `tokenizer/`, `files.json` over those two), plus `README.md` (model
   card: base model, Apache-2.0, `docs/RESULTS.md`, Swift usage) and `NOTICE`.
2. `create_repo(public, exist_ok=True)` and `upload_folder`; the upload's commit SHA is the `revision`.
3. Writes `swift/Sources/EmbeddingGemma/DefaultModel.swift` (repo, revision, sha256 of `files.json`) and commits it.
4. `git tag -a v1.0.0`, `gh repo edit --visibility public --accept-visibility-change-consequences`, pushes the
   branch and the tag, `gh release create v1.0.0 --notes-file docs/RESULTS.md`.

`--tag` overrides `v1.0.0`.

## Known failure modes

- **Fails after the upload**: the Hugging Face commit exists. Rerunning uploads again (a new commit) and pins that
  one; nothing else was changed unless the log shows the git steps ran.
- **s5, s6 and s7 show stale right after publishing**: step 3 commits `DefaultModel.swift`, and `swift/` is in their
  code paths. Expected; the published report records what was released.
- **An existing private Hugging Face repo stays private**: `exist_ok=True` does not change visibility. Make it
  public on the Hub, or use a new repo.
