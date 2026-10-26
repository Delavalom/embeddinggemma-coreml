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
4. Preflight: clean git tree, the tag does not exist, `docs/RESULTS.md` exists (Task 21's `gen_results`), and the
   two budget packages and archives s4's report says it shipped (`egcm.published.shipped`: `details.budgets`, recipe
   `details.winner`) are present, each with the functions of its budget.

Only then is `metrics.confirmed = true` set (the one gate, `[s7_publish].confirmed`).

## What it does

1. Builds `artifacts/s7_publish/upload/` (replaced), one folder per image budget (ADR-0010), `b70/` and `b280/`:
   - the app download: that budget's s4 device archive as is, `b<budget>/EmbeddingGemma2.aar` (LZFSE Apple Archive
     of `EmbeddingGemma2.mlpackage/**` and `tokenizer/{tokenizer.json,tokenizer_config.json}`, sha256-checked
     against s4's `checksums.json`), and `b<budget>/manifest.json` (`egcm.manifest`: format, archive, size, sha256,
     unpacked_size, image_budget, revision = the tag, functions = the text functions and that budget's image
     functions);
   - the plain layout of `egcm.published` for Python and other tools (the budget's package, uncompressed, as
     `b<budget>/EmbeddingGemma2.mlpackage/`; `b<budget>/tokenizer/`; `b<budget>/files.json` over those two);
   - at the root, `README.md`, `LICENSE` and `NOTICE` (Apache-2.0 section 4: recipients get the license text and
     the notices).
   `README.md` is the model card: `model_card.md` (a `string.Template`) filled by `card.py` from the reports
   (s0's revision and license, s1-s5 cosines, s4's chosen variant: size, ranking, latency, Neural Engine share,
   worst images, each budget's archive size, package size and unpacked size; s6's devices; every stage's `git_sha` and `code_hash`), with
   `docs/RESULTS.md` embedded. Edit the prose in `model_card.md`; never type a measured number into it. A
   placeholder without a value raises.
2. `create_repo(public, exist_ok=True)` and `upload_folder`; the upload's commit SHA is the `revision`.
3. Logs the upload's commit and the sha256 of each `b<budget>/manifest.json`, writes
   `swift/Sources/EmbeddingGemma/DefaultModel.swift` (repo, revision, `defaultManifestSHA256: [Int: String] =
   [70: "...", 280: "..."]`, one pin per budget) and commits it. The last stdout line is
   `PUBLISHED_MODEL {"repo", "revision", "manifest_sha256": {"70": ..., "280": ...}}`, what an app pins.
4. `git tag -a v1.0.0`, `gh repo edit --visibility public --accept-visibility-change-consequences`, pushes the
   branch and the tag, `gh release create v1.0.0` with notes = `docs/release-notes/<tag>.md` (hand-written, if
   present) followed by `docs/RESULTS.md`, written to `artifacts/s7_publish/release_notes.md`.

Before running it, go through `docs/publish-checklist.md`.

`--tag` overrides `v1.0.0`.

## Known failure modes

- **Fails after the upload**: the Hugging Face commit exists. Rerunning uploads again (a new commit) and pins that
  one; nothing else was changed unless the log shows the git steps ran.
- **s5, s6 and s7 show stale right after publishing**: step 3 commits `DefaultModel.swift`, and `swift/` is in their
  code paths. Expected; the published report records what was released.
- **An existing private Hugging Face repo stays private**: `exist_ok=True` does not change visibility. Make it
  public on the Hub, or use a new repo.
