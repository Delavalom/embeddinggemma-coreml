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
3. Changing `gates.toml` requires a new `docs/decisions/ADR-*.md` in the same commit. CI enforces this with `scripts/check_gates_adr.py` (planned: Task 21).
4. Fixtures only come from `python -m stages.s0_reference.build_fixtures` (planned: Task 5), which writes `fixtures/MANIFEST.sha256`. `just stage 0` refuses to run without that manifest, and CI checks it (planned: Task 21).
5. No weights or `.mlpackage` files in git. The `.gitignore` plus a CI size check (no file over 5 MB, `scripts/check_large_files.py`, planned: Task 21) enforce this.
6. Files stay under about 300 lines and have one job.
7. When a stage fails, read that stage's README "known failure modes" (planned: each stage's task) before changing code. Any new failure mode found gets added to it.

## Commands

All actions are one command. Run them from the repo root. Run scripts as modules (`python -m scripts.gen_status`), never by path.

```
just setup             # uv sync --group convert; swift package resolve once swift/ exists (planned: Task 16)
just status            # regenerate STATUS.md and print it
just stage <N>         # run stage N, write its report, exit non-zero on fail
just all               # stages 0–5 in order, stop at the first failure
just test              # pytest, plus swift test once swift/ exists (planned: Task 16)
just lint              # ruff check
just bench-device      # run the s6 benchmark on the connected iPhone (planned: Task 20)
just publish           # s7. Refuses to run unless s0–s6 pass AND --confirm is given (planned: Task 22)
```

`just stage N` needs `stages/sN_*/run.py` (planned: Task 6 for s0, then one per stage). Line 1 of each `run.py` is `# group: convert` or `# group: reference`; the recipe runs it with `uv run --exact --group <that group>`, then regenerates `STATUS.md` and keeps the stage's exit code.

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
- `min_cosine` in s4 and s5 is the minimum over text and image vectors combined. s1 to s3 gate text and image separately.
- Ranking metrics (`recall_at_k`, `spearman_topk`) take keyword arguments `ref=(queries, docs)` and `new=(queries, docs)`. s4 uses only the 120 app queries (`kind=query`), not the document-style texts.

## Provenance and staleness

Each report records where it came from:

- `git_sha`: `git rev-parse HEAD` at run time. `git_dirty` is true if any watched path differed from HEAD.
- `code_hash`: sha256 over `git rev-parse HEAD:<path>` for `egcm`, `stages/<stage>`, `gates.toml`, `pyproject.toml` and `uv.lock`. If those paths are dirty it also hashes `git diff HEAD -- <paths>` and the contents of untracked files in them.
- `inputs_sha256`: sha256 over the sorted `(stage, code_hash, inputs_sha256)` of each upstream report the stage reads, plus `fixtures/MANIFEST.sha256` (s0 only) and the Hugging Face revision `egcm.paths.HF_REVISION` (s0 only). Which upstream reports each stage reads is `STAGE_INPUTS` in `egcm/provenance.py`.

A report is **stale** (`egcm.provenance.is_stale`) if its `code_hash` differs from the current `code_hash(stage)`, or its `inputs_sha256` differs from the recomputed one, or any upstream report it reads (`STAGE_INPUTS`) is missing, not `passed`, or itself stale, checked recursively. `just status` shows it. A stale report does not count as passing. Changing code under `egcm/` or the lock file makes every stage stale; changing `stages/s0_reference/` makes s0 stale and, through the recursion, every stage after it. `run_stage` refuses to run (status `error`) when an upstream report is missing, not passed or stale, and says which stage to rerun, for example "s0_reference is stale (code changed); rerun `just stage 0`".

Run stages on a clean tree where you can: a report made on a dirty tree is stale as soon as you commit (the hash of the committed tree differs from the hash of the diff), so commit and then rerun.

## Stages

All thresholds live in `gates.toml`. Cosine values are per fixture, and the gate is on the **minimum**, not the mean. Ranking metrics use the 120 text queries against the images.

| Stage | What it does | Gate |
|---|---|---|
| **s0_reference** | Downloads the weights (revision pinned by commit SHA), runs the official `sentence-transformers` model in FP32 on CPU. Writes golden 768-d vectors, token ids, preprocessed pixel tensors and the exact image token sequence. Also lists every weight key and shape | deterministic: two runs give identical files. License recorded as Apache-2.0. Key inventory written |
| **s1_rewrite** | Our PyTorch rewrite, FP32, same inputs | min cosine ≥ 0.9999 (text and image). Strict weight load |
| **s2_coreml_fp32** | Convert at FP32, run on CPU (`.cpuOnly`) | min cosine ≥ 0.9999 |
| **s3_coreml_ane** | FP16, `.cpuAndNeuralEngine`, on this Mac's Neural Engine (M1 Pro) | min cosine ≥ 0.995, and ≥ 95% of ops placed on the Neural Engine (from `MLComputePlan`) |
| **s4_compress** | int8 weights first, then 4-bit palettization. Keep the smallest variant that passes | min cosine ≥ 0.99 (text and image combined), recall@10 ≥ 0.95, 5th-percentile Spearman ρ of the top-50 ≥ 0.9, full download ≤ 300 MB |
| **s5_swift** | Swift package on macOS: tokenizer, preprocessing, end-to-end vectors | token ids identical; pixel tensors max abs diff ≤ 1/255; end-to-end min cosine ≥ 0.99 against s0 |
| **s6_device** | PhotoSearchDemo on a real iPhone | records ms per image (70 and 280 budgets), ms per query, peak RSS, thermal state after 1,000 images. Gate: the report exists for ≥ 1 device; peak RSS ≤ 800 MB. Speed is recorded against the 60 ms target (`[s6_device.targets]`), not gated |
| **s7_publish** | HF upload, checksums, git tag `v1.0.0`, SwiftPM release, repo flipped to public | needs `--confirm` (given only after founder approval) |

## Key facts

- The model is rewritten in plain PyTorch in the Neural Engine layout (`(B, C, 1, S)`, 1x1 convs, per-head attention), loaded with Google's weights, then converted with `coremltools`. We do not trace Hugging Face's code.
- The model card says BF16 or FP32 only, not FP16. FP16 overflow is the main risk in s3; `egcm/probe.py` (planned: Task 14) finds where.
- Images use ten fixed shape buckets (budgets 70 and 280 x tall/portrait/square/landscape/wide). See `egcm/buckets.py` and `docs/decisions/ADR-0002-image-buckets.md`.
- `vendor/reference/` holds read-only upstream copies, pinned in `SOURCE.txt`. Read them for ground truth on the architecture; never import them.
- Do not guess model behavior. s0 records the real processor output (image token sequence, key inventory); build on that.

## Where things are

```
AGENTS.md  CLAUDE.md  LICENSE  NOTICE  justfile  gates.toml  pyproject.toml  uv.lock
README.md                 (planned)
STATUS.md                 GENERATED by `just status`. Never hand-edit
.github/workflows/ci.yml  (planned: Task 21)
vendor/reference/*.py     read-only upstream copies (+ SOURCE.txt with the commit)
egcm/                     shared Python library
  paths.py                filesystem locations, HF_REPO, HF_REVISION, STAGES
  report.py               Report schema and atomic read/write
  gates.py                loads gates.toml; evaluate(report, stage_key)
  provenance.py           code_hash, inputs_sha256, is_stale, STAGE_INPUTS
  stage.py                run_stage: the runner every stage uses
  metrics.py              cosine, recall@k, spearman
  buckets.py              bucket table, selection, resize-size math
  config.py               typed model config from HF config.json (planned: Task 7)
  model/                  norm.py rope.py attention.py text.py vision.py embedder.py (planned: Tasks 7–11)
  weights.py              safetensors -> rewrite mapping, strict (planned: Task 12)
  convert.py              torch -> mlpackage + multifunction merge (planned: Task 13)
  compress.py             int8 / palettize (planned: Task 15)
  probe.py                activation max-abs probe (planned: Task 14)
  fixtures.py             fixture manifest helpers (planned: Task 5)
stages/sN_*/              README.md (input, output, gate, known failure modes) + run.py (planned: one per stage)
scripts/                  package. gen_status.py now; gen_results.py, check_gates_adr.py, check_large_files.py (planned: Task 21)
tests/                    pytest unit tests for the Python library
fixtures/                 generated by s0; MANIFEST.sha256 is committed (planned: Task 5)
swift/                    SwiftPM package, Package.swift at swift/ (planned: Task 16)
Examples/PhotoSearchDemo/ SwiftUI demo app (planned: Task 20)
docs/decisions/           ADR-NNNN-*.md (numbered)
docs/RESULTS.md           GENERATED parity and benchmark tables (planned: Task 21)
docs/superpowers/         design spec and implementation plan
artifacts/                gitignored: model files, intermediate tensors, reports
```

The plan's file map is authoritative for anything marked planned.
