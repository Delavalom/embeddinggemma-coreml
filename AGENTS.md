# AGENTS.md

Read this first. It is written for an AI agent arriving with no prior context.

Items marked "(planned: Task N)" do not exist yet; they land with that task in `docs/superpowers/plans/2026-10-22-embeddinggemma-coreml.md`. Everything unmarked exists and is tested.

## What this repo is

A pipeline that converts Google's `google/embeddinggemma-2` (text + image embeddings) to Core ML so it runs on the iPhone Neural Engine, proves the result matches the original, and ships a Swift package (`EmbeddingGemma` + `EmbeddingIndex`) to use it.

The work is split into stages `s0`–`s7`. Each stage writes `artifacts/<stage>/report.json`, which is checked against thresholds in `gates.toml`. Nothing is "done" until its report says so.

Full design: `docs/superpowers/specs/2026-10-22-embeddinggemma-coreml-design.md`. Plan: `docs/superpowers/plans/2026-10-22-embeddinggemma-coreml.md`.

## Rules

1. Start every session with `just status`.
2. A stage only passes if its `artifacts/sN/report.json` has `"passed": true` and `"status": "passed"`, was produced by `just stage N`, and is not stale (see "Provenance and staleness"). The report records `git_sha`, `git_dirty`, `code_hash`, `inputs_sha256`, `gate_values`, `metrics`, `passed`, `status`, and `started_at` / `finished_at`.
3. Changing `gates.toml` requires a new `docs/decisions/ADR-*.md` in the same commit. CI enforces this with `scripts/check_gates_adr.py`.
4. Fixtures only come from `python -m stages.s0_reference.build_fixtures`, which writes `fixtures/MANIFEST.sha256`. `just stage 0` refuses to run without that manifest, and CI checks it.
5. No weights or `.mlpackage` files in git. The `.gitignore` plus a CI size check (no file over 5 MB, fixture images 2 MB, `scripts/check_large_files.py`) enforce this.
6. Files stay under about 300 lines and have one job.
7. When a stage fails, read that stage's README "known failure modes" (s0 has one; the others land with each stage's task) before changing code. Any new failure mode found gets added to it.

## Commands

All actions are one command. Run them from the repo root. Run scripts as modules (`python -m scripts.gen_status`), never by path.

```
just setup             # uv sync --group convert; swift package resolve
just status            # regenerate STATUS.md and print it
just results           # regenerate docs/RESULTS.md (gates, every s4 variant, s5 pixels, s6 devices)
just stage <N>         # run stage N, write its report, exit non-zero on fail
just all               # stages 0–5 in order, stop at the first failure
just test              # pytest, plus swift test (first run downloads the 32 MB tokenizer into artifacts/tokenizer/)
just lint              # ruff check
just bench-device      # run the s6 benchmark on the connected iPhone (bundles the s4 model into the demo)
just publish           # s7. Refuses unless --confirm, HF_REPO_TARGET, and s0–s6 passed and current. Never run it without founder approval
```

`just stage N` needs `stages/sN_*/run.py` (every stage has one). Line 1 of each `run.py` is `# group: convert` or `# group: reference`; the recipe runs it with `uv run --exact --group <that group>`, then regenerates `STATUS.md` and keeps the stage's exit code.

Python unit tests only: `uv run --exact --group convert pytest -q`.

## Python environments

- `convert` group: `torch` pinned to the version `coremltools` officially supports, plus `coremltools`, `pytest`. Used by s1–s4 and by all unit tests. Our rewrite never imports `transformers`.
- `reference` group: `transformers==5.20.2` (a release, so the environment is reproducible; `vendor/reference/` is copied from the same tag) plus `sentence-transformers`. Used only by s0.

They are declared as conflicting in `pyproject.toml`, so they cannot be installed together. `just setup` syncs only `convert`. The `reference` environment is installed the first time you run `just stage 0`, because `uv run --exact` makes the environment match one group exactly (and removes the other's packages). Switching between stages 0 and 1 therefore reinstalls packages from uv's cache.

## Reports, gates, and metrics

- Every stage uses `egcm.stage.run_stage(stage_key, fn)`. It writes a `running` report first (so an old passing report cannot survive a crash), stamps provenance, calls `fn(report)` which fills `report.metrics` and `report.details`, evaluates against `gates.toml`, writes atomically, and exits non-zero unless the stage passed. An exception in `fn` is recorded as status `error` with the traceback. Status is `running | passed | failed | error`.
- `gates.toml` has one table per stage, named by the full stage name (`[s1_rewrite]`). `egcm.gates.evaluate` takes its gates from the table of the report's own stage. A stage with no gates is an error.
- The comparison comes from the key: `min_*` is `>=`, `max_*` is `<=`, a bool is `==`. Any other key raises when `gates.toml` is loaded. **The prefix is the comparison direction only; each metric's docstring defines its aggregation** (for example `min_p5_spearman_top50` is the 5th percentile over queries, not a minimum).
- Recorded-only numbers (such as the s6 speed target) live in a `[stage.targets]` subtable. They are never gated; read them with `egcm.gates.targets(stage_key)`.
- A metric that is missing, non-numeric or non-finite is a recorded gate failure, not a crash. Non-finite floats are written to JSON as the strings `"NaN"`, `"Infinity"`, `"-Infinity"` and restored on read.
- `min_cosine` in s4 and s5 is the minimum over text and image vectors combined. s1 to s3 gate text and image separately. s5 compares against the FP reference on Swift's own pixels, and gates Swift vs Python Core ML on identical inputs as `min_cosine_runtime` (ADR-0008).
- Ranking metrics (`recall_at_k`, `spearman_topk`) take keyword arguments `ref=(queries, docs)` and `new=(queries, docs)`. s3 and s4 use only the 120 app queries (`kind=query`), not the document-style texts, through `egcm.ranking`; the reference for the gated ranking is s0's bucketed image vectors (ADR-0006, ADR-0007).

## Provenance and staleness

Each report records where it came from:

- `git_sha`: `git rev-parse HEAD` at run time. `git_dirty` is true if any watched path differed from HEAD.
- `code_hash`: sha256 over `git rev-parse HEAD:<path>` for the stage's code paths (`STAGE_CODE` in `egcm/provenance.py`: the stage's own directory, the `egcm` modules that decide its outputs, `pyproject.toml` and `uv.lock`), plus the stage's own table in `gates.toml` (by value, so other stages' tables, comments and layout do not count). If the code paths are dirty it also hashes `git diff HEAD -- <paths>` and the contents of untracked files in them. Infrastructure (`egcm/stage.py`, `provenance.py`, `report.py`, `redact.py`, `gates.py`, `scripts/`) is in no stage's paths: changing how reports are written does not change what a stage computes. When you add a module a stage uses, add it to that stage's `STAGE_CODE` entry.
- `inputs_sha256`: sha256 over the sorted `(stage, code_hash, inputs_sha256)` of each upstream report the stage reads, plus `fixtures/MANIFEST.sha256` (s0 only) and the Hugging Face revision `egcm.paths.HF_REVISION` (s0 only). Which upstream reports each stage reads is `STAGE_INPUTS` in `egcm/provenance.py`.

A report is **stale** (`egcm.provenance.is_stale`) if its `code_hash` differs from the current `code_hash(stage)`, or its `inputs_sha256` differs from the recomputed one, or any upstream report it reads (`STAGE_INPUTS`) is missing, not `passed`, or itself stale, checked recursively. `just status` shows it. A stale report does not count as passing. Changing the lock file makes every stage stale. Changing `egcm/model/` makes s1 to s4 stale but not s0. Changing `stages/s0_reference/` or a module s0 uses (`buckets.py`, `fixtures.py`, `paths.py`) makes s0 stale and, through the recursion, every stage after it. `run_stage` refuses to run (status `error`) when an upstream report is missing, not passed or stale, and says which stage to rerun, for example "s0_reference is stale (code changed); rerun `just stage 0`".

Run stages on a clean tree where you can: a report made on a dirty tree is stale as soon as you commit (the hash of the committed tree differs from the hash of the diff), so commit and then rerun.

## Stages

All thresholds live in `gates.toml`. Cosine values are per fixture, and the gate is on the **minimum**, not the mean. Ranking metrics use the 120 text queries against the images.

| Stage | What it does | Gate |
|---|---|---|
| **s0_reference** | Downloads the weights (revision pinned by commit SHA), runs the official `sentence-transformers` model in FP32 on CPU. Writes golden 768-d vectors, token ids, preprocessed pixel tensors and the exact image token sequence. Also lists every weight key and shape | deterministic: an in-process rerun of a sample (one image per bucket shape, a query, a document, the longest text) is bitwise identical. License recorded as Apache-2.0. Key inventory written. Enforced (raises): fixtures match the manifest, model sha256, `encode` agrees with the manual path, bit-exact pixel contract |
| **s1_rewrite** | Our PyTorch rewrite, FP32, same inputs | min cosine ≥ 0.9999 (text and image). Strict weight load |
| **s2_coreml_fp32** | Convert at FP32, run on CPU (`.cpuOnly`) | min cosine ≥ 0.9999 |
| **s3_coreml_ane** | FP16, `.cpuAndNeuralEngine`, on this Mac's Neural Engine (M1 Pro) | text min cosine ≥ 0.995; image min cosine ≥ 0.99 and 1st percentile ≥ 0.995; recall@10 ≥ 0.95 and p5 Spearman ≥ 0.9 against the bucketed FP reference (ADR-0006); ≥ 95% of ops placed on the Neural Engine (from `MLComputePlan`) |
| **s4_compress** | Linear weight quantization; ships `int8_table4` (int8 weights, int4 table; ADR-0007) | min cosine ≥ 0.99 (text and image combined); recall@10 ≥ 0.95 and 5th-percentile Spearman ρ of the top-50 ≥ 0.9 against the bucketed FP reference; exact-aspect top-1 in the top 10 ≥ 0.95 and image→image category precision@10 drop ≤ 0.03 (bucketing); device download (Apple Archive, LZFSE, with the tokenizer) ≤ 360 MB |
| **s5_swift** | Swift package on macOS: tokenizer, preprocessing, end-to-end vectors | token ids identical; pixel tensors max abs diff ≤ 1/255 (p99 on decode-identical inputs, ADR-0004); Swift vs Python Core ML on identical inputs min cosine ≥ 0.9999; s4's task-level gates (min cosine ≥ 0.99, recall@10, p5 Spearman, bucketing) against the FP reference on Swift's pixels (ADR-0008). Against s0 it is recorded only |
| **s6_device** | PhotoSearchDemo on a real iPhone | records ms per image (70 and 280 budgets), ms per query, peak RSS, thermal state after 1,000 images. Gate: the report exists for ≥ 1 device; peak RSS ≤ 800 MB. Speed is recorded against the 60 ms target (`[s6_device.targets]`), not gated |
| **s7_publish** | HF upload, checksums, git tag `v1.0.0`, SwiftPM release, repo flipped to public | needs `--confirm` (given only after founder approval) |

## Key facts

- The model is rewritten in plain PyTorch in the Neural Engine layout (`(B, C, 1, S)`, 1x1 convs, per-head attention), loaded with Google's weights, then converted with `coremltools`. We do not trace Hugging Face's code.
- The model card says BF16 or FP32 only, not FP16. FP16 overflow is the main risk in s3; `egcm/probe.py` records the FP32 activation ranges (s3).
- Fixture pixels are sRGB. Downstream stages read s0's bucket images only through `egcm.fixtures.load_bucket_pixels(budget, image_id)` (`u8.astype(np.float32) * np.float32(1/255)`, `[1, 3, H, W]`).
- Images use ten fixed shape buckets (budgets 70 and 280 x tall/portrait/square/landscape/wide). See `egcm/buckets.py` and `docs/decisions/ADR-0002-image-buckets.md`.
- `vendor/reference/` holds read-only upstream copies, pinned in `SOURCE.txt`. Read them for ground truth on the architecture; never import them.
- Do not guess model behavior. s0 records the real processor output (image token sequence, key inventory); build on that.

## Where things are

```
AGENTS.md  CLAUDE.md  LICENSE  NOTICE  justfile  gates.toml  pyproject.toml  uv.lock
Package.swift             the SwiftPM manifest consumers resolve (targets at path: swift/Sources/*); keep in step with swift/Package.swift
README.md                 (planned)
STATUS.md                 GENERATED by `just status`. Never hand-edit
.github/workflows/ci.yml  macos-latest: pytest, gates-ADR check, large files, ruff, fixture manifest, swift build/test
vendor/reference/*.py     read-only upstream copies (+ SOURCE.txt with the commit)
egcm/                     shared Python library
  paths.py                filesystem locations, HF_REPO, HF_REVISION, STAGES
  report.py               Report schema and atomic read/write
  gates.py                loads gates.toml; evaluate(report, stage_key)
  provenance.py           code_hash, inputs_sha256, is_stale, STAGE_INPUTS
  stage.py                run_stage: the runner every stage uses (deletes its Core ML E5 bundles on exit)
  e5exit.py               deletes the E5 cache bundles a process created when it exits (stdlib only)
  metrics.py              cosine, recall@k, spearman
  buckets.py              bucket table, selection, resize-size math
  config.py               typed model config from HF config.json; checks the assumptions the rewrite relies on
  model/                  norm.py (RMSNormCF) rope.py (1-D / axial 2-D tables) attention.py (AttentionCF)
                          masks.py (key_mask); text.py (TextTower) vision.py (VisionEncoder, VisionTower per
                          bucket) embedder.py (TextEmbedder, ImageEmbedder: the Core ML entry points)
  weights.py              safetensors -> rewrite mapping, strict load, q.k logit bound check
  functions.py            the spec §7 entry points: names, I/O contract (int32 mask), torch module per function
  convert.py              torch -> mlpackage, batched multifunction merge (weights stored once), contract check
  evaluate.py             runs every s0 fixture through a compiled package (s2-s4), latency
  parts.py                builds packages one process per part and per merge (16 GB Mac), resumable, memory/disk guards
  compress.py             s4 recipes: deterministic linear quantization (int8 / int4 per block), table separately
  archive.py              the device download: one Apple Archive (LZFSE) of the package + tokenizer (s4 gate, s7)
  ranking.py              task-level metrics (s3, s4, s5): cosines, ranking vs bucketed FP, bucketing fidelity
  probe.py                activation max-abs probe (s3, diagnostic)
  placement.py            Neural Engine placement per function from MLComputePlan (s3, s4)
  fixtures.py             fixture loading and manifest helpers (load_texts, load_images, verify_manifest)
  published.py            what s4 shipped (shipped(): package, device archive, recipe, functions, from its report) and the
                          plain published folder (files.json + mlpackage + tokenizer): s6 bundles it, s7 uploads it
  manifest.py             manifest.json for the device archive (the app/Roll contract), unpack + layout check (s5-s7)
  redact.py               repo-relative paths, no home folder, in reports and anything published (report.py, s7)
stages/sN_*/              README.md (input, output, gate, known failure modes) + run.py
scripts/                  package. gen_status.py, gen_results.py, check_gates_adr.py, check_large_files.py,
                          save_result.py (a report into docs/results/, paths relativized)
tests/                    pytest unit tests for the Python library
fixtures/                 written by build_fixtures and committed: images/, images.jsonl, texts.jsonl (hand-written), commons_sources.json, MANIFEST.sha256
swift/                    SwiftPM package (Package.swift here). EmbeddingGemma: Buckets, ImagePreprocessor + Resampler
                          (torch uint8 AA bicubic port), TextEncoder, ModelStore (files.json, SHA-256, compile),
                          DeviceTier, the EmbeddingGemma actor. EmbeddingIndex: VectorIndex. Tests/.../Fixtures from
                          stages/s5_swift/export_fixtures.py; RuntimeTests need make_contract_model, EndToEndTests s4
                          egcm-parity: the s5 CLI (stages/s5_swift/run.py runs it; `--measure <pkg>` without a report;
                          `--pixels <dir>` dumps the bucket images, EgcmParityTests)
Examples/PhotoSearchDemo/ SwiftUI demo app + s6 benchmark (XcodeGen project.yml; Team.xcconfig and Model/ gitignored)
docs/decisions/           ADR-NNNN-*.md (numbered)
docs/RESULTS.md           GENERATED by `just results`: parity and benchmark tables. Never hand-edit
docs/superpowers/         design spec and implementation plan
artifacts/                gitignored: model files, intermediate tensors, reports
```

The plan's file map is authoritative for anything marked planned.
