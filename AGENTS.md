# AGENTS.md

Read this first. It is written for an AI agent arriving with no prior context.

## What this repo is

A pipeline that converts Google's `google/embeddinggemma-2` (text + image embeddings) to Core ML so it runs on the iPhone Neural Engine, proves the result matches the original, and ships a Swift package (`EmbeddingGemma` + `EmbeddingIndex`) to use it.

The work is split into stages `s0`–`s7`. Each stage writes `artifacts/<stage>/report.json`, which is checked against thresholds in `gates.toml`. Nothing is "done" until its report says so.

Full design: `docs/superpowers/specs/2026-10-22-embeddinggemma-coreml-design.md`. Plan: `docs/superpowers/plans/2026-10-22-embeddinggemma-coreml.md`.

## Rules

1. Start every session with `just status`.
2. A stage only passes if its `artifacts/sN/report.json` has `"pass": true` and was produced by `just stage N` at the current git SHA. The report records `git_sha`, `inputs_sha256`, `gate_values`, `metrics`, `pass`, and `started_at` / `finished_at`.
3. Changing `gates.toml` requires a new `docs/decisions/ADR-*.md` in the same commit. CI enforces this with `scripts/check_gates_adr.py`.
4. Fixtures only come from `just stage 0`. CI checks `fixtures/MANIFEST.sha256`.
5. No weights or `.mlpackage` files in git. The `.gitignore` plus a CI size check (no file over 5 MB) enforce this.
6. Files stay under about 300 lines and have one job.
7. When a stage fails, read that stage's README "known failure modes" before changing code. Any new failure mode found gets added to it.

Note on the report field: the JSON key is `pass` in the rules above, and the Python model (`egcm/report.py`) exposes it as `passed` with a `failures` list. Read `egcm/report.py` for the exact schema.

## Commands

All actions are one command. Run them from the repo root.

```
just setup             # uv sync both groups, swift package resolve
just status            # regenerate STATUS.md and print it
just stage <N>         # run stage N, write its report, exit non-zero on fail
just all               # stages 0–5 in order, stop at the first failure
just test              # pytest + swift test
just bench-device      # build PhotoSearchDemo to the connected iPhone, run the s6 benchmark
just publish           # s7. Refuses to run unless s0–s6 pass AND --confirm is given
```

Python unit tests only: `uv run --group convert pytest -q`.

Tools: `uv` (Python 3.12), `just`, Swift/Xcode. Python deps are in two groups that cannot be installed together (see below).

## Python environments

- `convert` group: `torch` pinned to the version `coremltools` officially supports, plus `coremltools`, `pytest`. Used by s1–s4 and by all unit tests. Our rewrite never imports `transformers`.
- `reference` group: `transformers` from git main plus `sentence-transformers`. Used only by s0.

They are declared as conflicting in `pyproject.toml`. The first line of each stage's `run.py` is `# group: <name>`, and the `justfile` uses it to pick the environment.

## Stages

All thresholds live in `gates.toml`. Cosine values are per fixture, and the gate is on the **minimum**, not the mean. Ranking metrics use the 200 text queries against the 200 images.

| Stage | What it does | Gate |
|---|---|---|
| **s0_reference** | Downloads the weights (revision pinned by commit SHA), runs the official `sentence-transformers` model in FP32 on CPU. Writes golden 768-d vectors, token ids, preprocessed pixel tensors and the exact image token sequence. Also lists every weight key and shape | deterministic: two runs give identical files. License recorded as Apache-2.0. Key inventory written |
| **s1_rewrite** | Our PyTorch rewrite, FP32, same inputs | min cosine ≥ 0.9999 (text and image). Strict weight load |
| **s2_coreml_fp32** | Convert at FP32, run on CPU (`.cpuOnly`) | min cosine ≥ 0.9999 |
| **s3_coreml_ane** | FP16, `.cpuAndNeuralEngine`, on this Mac's Neural Engine (M1 Pro) | min cosine ≥ 0.995, and ≥ 95% of ops placed on the Neural Engine (from `MLComputePlan`) |
| **s4_compress** | int8 weights first, then 4-bit palettization. Keep the smallest variant that passes | min cosine ≥ 0.99, recall@10 ≥ 0.95, Spearman ρ of the top-50 ≥ 0.9, full download ≤ 300 MB |
| **s5_swift** | Swift package on macOS: tokenizer, preprocessing, end-to-end vectors | token ids identical; pixel tensors max abs diff ≤ 1/255; end-to-end min cosine ≥ 0.99 against s0 |
| **s6_device** | PhotoSearchDemo on a real iPhone | records ms per image (70 and 280 budgets), ms per query, peak RSS, thermal state after 1,000 images. Gate: the report exists for ≥ 1 device; peak RSS ≤ 800 MB. Speed is recorded against the 60 ms target, not gated |
| **s7_publish** | HF upload, checksums, git tag `v1.0.0`, SwiftPM release, repo flipped to public | needs `--confirm` (given only after founder approval) |

## Key facts

- The model is rewritten in plain PyTorch in the Neural Engine layout (`(B, C, 1, S)`, 1x1 convs, per-head attention), loaded with Google's weights, then converted with `coremltools`. We do not trace Hugging Face's code.
- The model card says BF16 or FP32 only, not FP16. FP16 overflow is the main risk in s3; `egcm/probe.py` finds where.
- Images use six fixed shape buckets (70 and 280 token budgets x portrait/landscape/square). See `egcm/buckets.py` and `docs/decisions/ADR-0002-image-buckets.md`.
- `vendor/reference/` holds read-only upstream copies, pinned in `SOURCE.txt`. Read them for ground truth on the architecture; never import them.
- Do not guess model behavior. s0 records the real processor output (image token sequence, key inventory); build on that.

## Where things are

```
AGENTS.md  CLAUDE.md  README.md  LICENSE  NOTICE  justfile  gates.toml  pyproject.toml
STATUS.md                 GENERATED by `just status`. Never hand-edit
.github/workflows/ci.yml
vendor/reference/*.py     read-only upstream copies (+ SOURCE.txt with the commit)
egcm/                     shared Python library
  paths.py                all filesystem locations
  report.py               Report schema, write/read, gate evaluation
  gates.py                loads gates.toml
  metrics.py              cosine, recall@k, spearman
  config.py               typed model config from HF config.json
  buckets.py              bucket table, selection, resize-size math
  model/                  norm.py rope.py attention.py text.py vision.py embedder.py
  weights.py              safetensors -> rewrite mapping (strict)
  convert.py              torch -> mlpackage (+ multifunction merge)
  compress.py             int8 / palettize
  probe.py                activation max-abs probe
  fixtures.py             fixture manifest helpers
stages/sN_*/              README.md (input, output, gate, known failure modes) + run.py
scripts/                  gen_status.py gen_results.py check_gates_adr.py check_large_files.py
tests/                    pytest unit tests for the Python library
fixtures/                 generated by s0; MANIFEST.sha256 is committed
swift/                    SwiftPM package (Package.swift at swift/)
Examples/PhotoSearchDemo/ SwiftUI demo app
docs/decisions/           ADR-NNNN-*.md (numbered)
docs/RESULTS.md           GENERATED parity and benchmark tables
docs/superpowers/         design spec and implementation plan
artifacts/                gitignored: model files, intermediate tensors, reports
```

Some of these do not exist yet; they are created as the plan's tasks land. The plan's file map is authoritative.
