# Pre-publish checklist (v1.0.0)

Go through this list before `HF_REPO_TARGET=delavalom/embeddinggemma-2-coreml just publish --confirm`. That one command does every
public step in order: Hugging Face upload, pinning `DefaultModel.swift`, the tag, the repo visibility flip, the
push and the GitHub release (`stages/s7_publish/README.md`). **The founder's approval for it covers all of those
steps at once.** If they should be approved separately, do not run s7; do the steps by hand in its order instead.

Items marked **[founder]** need explicit founder approval. Nothing else here is irreversible.

## 0. Blockers found while preparing (2026-10-25)

- [x] **SPM can install the package.** A root `Package.swift` (resolved 2026-10-25 on `integrate/s5-s7`) declares
      the same products, platforms and dependency with `path: "swift/Sources/..."`; `swift/Package.swift` stays for
      local `swift test` and `egcm-parity`. It is outside `swift/`, so the s5-s7 code hashes are unaffected.
      Checked with `swift package describe` and `swift build` at the repo root (CI builds it too). Keep the two
      manifests in step.
- [x] **The Hugging Face repo name is one value.** `stages/s7_publish/run.py` `DOCUMENTED_HF_REPO` =
      `delavalom/embeddinggemma-2-coreml`, which `README.md` and `docs/swift-package.md` name (a test checks it).
      s7's preflight refuses an `HF_REPO_TARGET` those docs do not name. The model card takes the repo from s7.
      s7 then uses the repo id exactly as the Hub returns it (case included) and prints
      `PUBLISHED_MODEL {"repo", "revision", "manifest_sha256"}`: the three values apps pin. Create the HF repo
      **public**, or let s7 create it: an existing private repo stays private (`exist_ok=True`).
- [ ] **Run s7 from the release branch.** `docs/publish` is merged into `integrate/s5-s7`. s7 pushes `HEAD` and
      tags it, so run it on the branch that should become public (`main` after the merge), with a clean tree.

## 1. Stage gates

- [ ] `just status`: s0-s6 are all `passed` and not stale (s7 refuses otherwise).
- [ ] s6 has run on at least one iPhone, and `docs/results/s6_<model>.json` is committed (s6 README).
- [ ] `just results` was run after the last stage, and `docs/RESULTS.md` is committed (s7 requires it; it is the
      body of the release notes and is embedded in the model card).
- [ ] `artifacts/s4_compress/report.json`: `details.winner` is `int8_table4`, not `none`. A non-winner package
      would be "chosen" but not shippable, and s4 would not have passed.

## 2. License and notice files

- [x] `LICENSE` (Apache 2.0 full text) is at the repo root. s7 now copies it into the Hugging Face upload
      (Apache 2.0 section 4(a)).
- [x] `NOTICE` lists the upstream model (revision, Google DeepMind, Apache 2.0), states how the weights were
      modified (section 4(b)), quotes the upstream Gemma Prohibited Use Policy requirement, and lists the vendored
      and modified Transformers code. s7 copies it into the upload (section 4(d)).
- [x] The upstream license is verified as Apache 2.0 at `HF_REVISION` 914f7f89142e33e77833254d9c9b90c3cef7303b. The
      card front matter says `license: apache-2.0`, and the link points to
      `ai.google.dev/gemma/docs/gemma_4_license`, which is the Apache 2.0 text. s0 recorded `license: apache-2.0`
      and its `license_apache` gate passed. **It is not the Gemma Terms of Use.** (That covered EmbeddingGemma 1;
      do not copy its terms here.) The only use restriction upstream is the model card's sentence "Deployments
      must adhere to the Gemma Prohibited Use Policy". It is passed through in the model card, `NOTICE`,
      `README.md` and `docs/swift-package.md`.
- [ ] Re-check the upstream card at `HF_REVISION` if the revision is ever bumped (`artifacts/hf/.../README.md`;
      s0's gate checks only the front matter).
- [x] Fixture images are CC0 / public domain (95 CC0, 5 PD from Commons), plus synthetic images
      (`fixtures/README.md`). No attribution file is required. Their removal-on-request note is in the README.

## 3. Secrets, emails and private data in the public tree and history

Checked on 2026-10-25 over `feat/s6-s7`, `integrate/s5-s7`, `feat/rewrite`, `feat/s5-ci` and all history
(`git grep` for email addresses, `hf_`/`ghp_`/`github_pat_`/`sk-`/AWS keys/private-key headers/`api_key=`/
`password=`/`token=`, absolute `/Users/` paths, a committed `DEVELOPMENT_TEAM`; `git rev-list --objects --all`
for large or secret-looking files):

- [x] No tokens, keys or credentials in any tracked file or in history. `Team.xcconfig` (signing team) is
      gitignored. Only a fake `ABCDE12345` appears, in a test.
- [x] Emails in tracked files: only `t@example.com` (`tests/conftest.py`) and `me@example.org`
      (`tests/test_commons.py`), both test placeholders. No real address.
- [x] Commit metadata: every author and committer is `Delavalom <hi@delavalom.com>`, the allowed contact. 104
      commit messages carry `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. That is intended
      attribution, not a leak.
- [x] No tracked file contains an absolute home path.
- [x] Reports store repo-relative paths: `egcm.report.write_report` rewrites paths under the repo as relative and
      the home folder as `~` (`egcm.redact`). Reports written before that still hold absolute paths, so copy a
      report into `docs/results/` only with `python -m scripts.save_result <stage>`, which relativizes and refuses
      a home path. s7's preflight refuses `docs/RESULTS.md` or `docs/results/*.json` with a home path, and
      `build_upload` refuses an upload that contains one. `tests/test_redact.py` fails on a committed one.
- [ ] Rerun right before the flip, on the branch being published:
      `git grep -n -I -E "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}" -- . ':!uv.lock' | grep -v -E "example\.(com|org)"`
      and `git log --all --format='%ae %ce' | tr ' ' '\n' | sort -u` (expect only `hi@delavalom.com`).

## 4. Artifact checksums

- [ ] `artifacts/s4_compress/checksums.json` (written only when s4 has a winner) has the device archive's sha256
      and byte count (`"archive"`), plus the sha256 of every file in the package. s7 uploads that archive as is
      (`egcm.manifest.write` refuses it if its sha256 differs), so `upload/manifest.json` `sha256` and `size` equal
      `checksums.json` `"archive"`. Compare each `EmbeddingGemma2.mlpackage/...` entry of `upload/files.json` with
      `checksums.json` too: they must be equal, because `published.assemble` clones the package byte for byte.
- [ ] The tokenizer entries in `files.json` equal `egcm.published.TOKENIZER_FILES` (`assemble` checks this and
      raises otherwise).
- [ ] After the upload, `DefaultModel.swift` has `defaultRepo` = the Hub's repo id, `defaultRevision` = the Hugging
      Face commit SHA from the s7 report (`details.revision`), and `defaultManifestSHA256` =
      `details.manifest_json_sha256` = sha256 of the uploaded `manifest.json`, the same three values as s7's
      `PUBLISHED_MODEL` line (give those to Roll). Check:
      `curl -sL https://huggingface.co/<repo>/resolve/<revision>/manifest.json | shasum -a 256`.
- [ ] Smoke test on a clean device or simulator: `EmbeddingGemma.load(.huggingFace(revision: defaultRevision))`
      downloads, verifies and embeds. A checksum mismatch means the pin is wrong.

## 5. Model card numbers match the reports

The card is rendered from the reports (`stages/s7_publish/card.py`, template `model_card.md`). Measured numbers
are never typed into the template. Preview it without publishing:

```
uv run --exact --group convert python -c "from pathlib import Path; from stages.s7_publish import run as s7; \
print(s7.model_card('<owner>/<name>', Path('docs/RESULTS.md').read_text()))" > /tmp/model_card.md
```

- [ ] No `pending` in the preview. s0-s6 passing means every value is present. A `pending` means a key moved:
      fix `card.py`, not the template.
- [ ] Spot-check against `artifacts/s4_compress/report.json` `details.variants.int8_table4`. The ADR-0007
      measurement (`measure/fast.json`, commit 77d1241) gave: package zip 349.7 MB (the archive replaced the zip as the
      gated download in 7e630f3), package 452.4 MB, min cosine text
      0.99791 / image 0.99133, recall@10 vs bucketed FP 0.962 / 0.966, p5 Spearman 0.966 / 0.972, `embed_text_16`
      4.4 ms, `embed_image_70_square` 45.8 ms, Neural Engine share 99.81%. The stage run should be close to
      these, not necessarily identical.
- [ ] Spot-check the iPhone table against `artifacts/s6_device/report.json` `details.devices`.
- [ ] The template has some fixed text that does not come from a report. Check it is still true:
      - ADR-0006's historical s3 numbers: i141 / i161 at 0.99404 / 0.99405, the other 398 at 0.998 or above,
        GPU 2.8x slower at budget 70, recall@10 0.9975 / 0.9992 with only those two swapped in.
      - "the v1.0.0 build Mac is an M1 Pro with macOS 26": no report records the host. Confirm it is the machine
        that ran s4.
      - The gate values quoted in the prose (800 MB, 60 ms target, 0.99 / 0.95 / 0.9, 0.95 / 0.03). The download
        limit and every size (archive, unpacked, package) come from the s4 report and `manifest.json`.

## 6. The public steps (all done by `just publish --confirm`)

- [ ] **[founder]** Hugging Face upload: `create_repo(public)` + `upload_folder` of
      `artifacts/s7_publish/upload/` (`manifest.json` + `EmbeddingGemma2.aar` for apps; the unpacked model,
      tokenizer and `files.json` for Python; `README.md`, `LICENSE`, `NOTICE`).
- [ ] **[founder]** Swift package release: s7 commits `DefaultModel.swift`, creates `git tag -a v1.0.0`, pushes
      the branch and the tag, and runs `gh release create v1.0.0`. The notes are `docs/release-notes/v1.0.0.md`
      followed by `docs/RESULTS.md`.
- [ ] **[founder]** GitHub repo visibility flip: `gh repo edit --visibility public
      --accept-visibility-change-consequences` on `Delavalom/embeddinggemma-coreml` (currently PRIVATE). Every
      branch and all history become public, not just `main`. Delete branches that should not be public first.
- [ ] After publishing: s5, s6 and s7 show stale (the pin commit touches `swift/`). That is expected (s7 README).
