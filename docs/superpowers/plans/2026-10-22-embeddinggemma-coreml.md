# EmbeddingGemma 2 → Core ML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a verified Core ML conversion of `google/embeddinggemma-2` (text + image) for the iPhone Neural Engine, plus a Swift package to embed and search.

**Architecture:**
- A staged pipeline (s0–s7). Each stage writes a `report.json` checked against `gates.toml`.
- A plain-PyTorch, Neural-Engine-shaped rewrite of the text and vision towers, loaded from Google's safetensors and converted with coremltools into one multifunction `.mlpackage`.
- A SwiftPM package that runs it.

**Tech stack:** Python 3.12 + uv, torch, coremltools, safetensors, numpy, pydantic, pytest; `transformers==5.20.2` + `sentence-transformers>=6.1` (reference only). Swift 6, SwiftPM, swift-transformers (Tokenizers), Accelerate, Core ML. `just`, GitHub Actions (macOS).

**Spec:** `docs/superpowers/specs/2026-10-22-embeddinggemma-coreml-design.md`. Read it first.

**Reference source:** the upstream files below, copied into `vendor/reference/` by Task 1 for reading only (Apache-2.0; never imported):
- `transformers/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py`
- `transformers/src/transformers/models/gemma4/modeling_gemma4.py` (classes `Gemma4Vision*`, `Gemma4RMSNorm`, `Gemma4ClippableLinear`)
- `transformers/src/transformers/models/gemma4/image_processing_gemma4.py`

---

## Facts that drive the design (checked against the reference source on 2026-10-22)

**Text tower** (`EmbeddingGemma2TextModel`), with `hidden=512`, `layers=24`, `heads=4`:
- **Embeddings:** `embed_tokens(ids) * sqrt(512)`. The BF16 rounding of the scale doesn't matter in FP32; we use `512**0.5`.
- **PLE (per-layer input):** `ple = RMSNorm_512( reshape( Linear(512→24·512)(emb) * 512**-0.5 , [S,24,512]) )`. It is computed from `inputs_embeds` only, with **no token lookup** (risk R4 is closed).
- **Layer i:**
  1. `h = h + post_attn_norm(attn(input_norm(h)))`
  2. `h = h + post_ff_norm(mlp(pre_ff_norm(h)))`
  3. `h = h + post_ple_norm(ple_proj(gelu_tanh(ple_gate(h)) * ple[:, :, i]))`
  4. `h = h * layer_scalar[i]` (a persistent buffer, loaded from the weights)
- **Attention:**
  - Bidirectional on every layer, `scaling = 1.0`.
  - `q_norm`/`k_norm` RMSNorms with weights; `v_norm` RMSNorm without weights. Norms are applied per head *before* RoPE.
  - RoPE uses `rotate_half` on the full head_dim.
- **Layer types:** sliding layers have head_dim 256, 2 KV heads, θ 1e4. Full layers (5, 11, 17, 23) have head_dim 512, 1 KV head, θ 1e6.
  - The sliding window is 512. Every sequence we build is ≤ `egcm.buckets.MAX_SEQ` = 284 tokens (280 soft tokens + 4 special), so sliding layers attend to everything: **one mask for all layers**. `config.load()` rejects a config whose `sliding_window` is below `MAX_SEQ`.
- **Output:** `norm(h)` → `embedding_projection` Linear(512→768) per token → mean over valid tokens (sentence-transformers `include_prompt: true`) → L2 normalize.
- **RMSNorm:** `x * (mean(x²) + eps)^-0.5 * weight` in FP32. **Gemma4 weights are used as-is** (no `1 + w`): `weight` is initialized to ones and multiplied directly.

**Vision tower** (`Gemma4VisionModel`), with `hidden=768`, `16 layers`, `12 heads`, head_dim 64, patch 16, pooling k=3:
- **Patch embedding:** `input_proj( 2*(patches-0.5) )` plus position embeddings `table[0][x] + table[1][y]`. Patches are flattened in `(py, px, c)` order (`permute(1,3,2,4,0)`).
- **RoPE:** 2-D axial, θ 100, head_dim 64.
  - `inv_freq = 1/100^(arange(0,32,2)/32)`, which gives 16 frequencies.
  - Per position: `cos = cat[cos(x·f), cos(x·f), cos(y·f), cos(y·f)]` (64 values). `apply_multidimensional_rope` splits the head into two 32-channel halves (x half, y half) and applies `rotate_half` within each.
  - QK/V norms are as in text; `scaling = 1.0`; bidirectional.
- **Clipped linears:** `use_clipped_linears = False` for vision, so they are plain linears. Weight keys are `...q_proj.linear.weight`.
- **Pooler:** zero the padding → average each 3×3 patch block (row-major over blocks) → `* sqrt(768)` in FP32 → (standardize = False).
- **`embed_vision`:** `RMSNorm_768(no weight)` → Linear(768→512). RMSNorm without weight is scale-invariant, so **the rewrite drops the `* sqrt(768)`** (an exact identity apart from eps, ~1e-6 relative). This removes the one FP16 overflow upstream had to call out.
- **Image sequence (recorded by s0 in `image_token_layout.json`, identical in shape for all ten buckets):** `[<bos>=2, <|image>=255999, <|image|>=258880 × soft_tokens, <image|>=258882, <eos>=1]`, so the length is soft_tokens + 4 (64 to 284 tokens). The soft tokens replace the `<|image|>` placeholders as `embed_vision` outputs, **unscaled**; the other four tokens are `embed_tokens(id) * sqrt(512)` (exactly `512**0.5` in FP32, checked against `debug.npz`). PLE is computed from this merged `inputs_embeds`. Mean pooling covers all tokens, including bos, eos, `<|image>` and `<image|>`.
- **Text sequence (recorded by s0 in `texts.npz`):** `[<bos>=2] + tokens(prompt + text) + [<eos>=1]`, with the prompt `task: search result | query: ` (queries) or `title: none | text: ` (documents) concatenated as plain text. The longest fixture text (t199) is exactly 128 tokens, so every fixture text fits `embed_text_128`.
- **Processor patches:** `pixel_values` are padded to `budget·9` rows with position `(-1, -1)`. The valid rows equal, bit for bit, the bucket image `u8.astype(float32) * float32(1/255)` patchified in `(py, px, c)` order (not `u8 / 255`, which differs by up to 6e-8), and the positions are the row-major `(x, y)` meshgrid. s0 asserts both and stores the bucket image as uint8 HxWx3; read it with `egcm.fixtures.load_bucket_pixels`.

**Image preprocessing:**
1. Convert to RGB, then resize with bicubic + antialias (`resample: 3`) to the largest size that keeps aspect ratio, whose sides are multiples of 48, and whose area is ≤ budget·9·256.
2. Rescale by 1/255; no normalization.

**Buckets (exact fixed points of that resize, so the reference and our model see identical pixels).** Ten buckets: five aspect ratios for each budget. `docs/decisions/ADR-0002-image-buckets.md` has the patch grids, stretch figures and rationale.

| Budget | Bucket | Size (H×W px) | Soft tokens |
|---|---|---|---|
| 70 | tall | 576×240 | 12×5 = 60 |
| 70 | portrait | 480×336 | 10×7 = 70 |
| 70 | square | 384×384 | 8×8 = 64 |
| 70 | landscape | 336×480 | 7×10 = 70 |
| 70 | wide | 240×576 | 5×12 = 60 |
| 280 | tall | 1152×528 | 24×11 = 264 |
| 280 | portrait | 960×672 | 20×14 = 280 |
| 280 | square | 768×768 | 16×16 = 256 |
| 280 | landscape | 672×960 | 14×20 = 280 |
| 280 | wide | 528×1152 | 11×24 = 264 |

**Bucket choice from the aspect ratio `r = W/H`:** `r < 0.6` → tall; `0.6 ≤ r < 0.83` → portrait; `0.83 ≤ r ≤ 1.2` → square; `1.2 < r ≤ 1.67` → landscape; `r > 1.67` → wide. `choose_bucket(height, width, budget)` takes the height first.

## Neural Engine rewrite rules (from Apple's `ml-ane-transformers`)

- **Layout:** hidden states are `(1, C, 1, S)`, and every Linear is a `Conv2d(1×1)`.
- **RMSNorm:** reduces over the channel dim (1), scale-safe form: `x/s · rsqrt(mean((x/s)²) + eps/s²)`, where `s` is a per-norm constant (default 1, set from the probe in s3).
- **Attention:** per head, using `einsum('bchq,bkhc->bkhq')`; no 5-D tensors. Softmax over dim 1.
- **Vision patchify:** `Conv2d(3→768, k=16, s=16)`, with the weight built from `input_proj.weight.view(768,16,16,3).permute(0,3,1,2) * 2` and the bias set to `-input_proj.weight.sum(1)`. This folds in `2(x−0.5)`.
- **Vision position embeddings and every RoPE cos/sin:** precomputed per bucket and stored as constants.
- **Vision pooling:** `avg_pool2d(k=3, s=3)` on the `(1,768,Hp,Wp)` grid → `(1,768,1,T)`.

---

## File map

```
embeddinggemma-coreml/
  AGENTS.md  CLAUDE.md  README.md  LICENSE(Apache-2.0)  NOTICE  .gitignore  justfile  gates.toml  pyproject.toml
  .github/workflows/ci.yml
  vendor/reference/*.py                 # read-only upstream copies (Task 1)
  egcm/__init__.py
  egcm/paths.py                         # all filesystem locations
  egcm/report.py                        # Report schema, atomic write/read (no gate logic)
  egcm/gates.py                         # load gates.toml (gates, targets), evaluate(report, stage_key)
  egcm/provenance.py                    # git sha, code_hash, inputs_sha256, is_stale, STAGE_INPUTS
  egcm/stage.py                         # run_stage(stage_key, fn): the runner every stage uses
  egcm/metrics.py                       # cosine, recall@k, spearman
  egcm/config.py                        # typed model config from HF config.json
  egcm/buckets.py                       # bucket table + selection + resize-size math
  egcm/model/norm.py                    # RMSNorm (channels-first, scale-safe)
  egcm/model/rope.py                    # 1-D and 2-D axial rope tables
  egcm/model/attention.py               # ANE attention (text and vision)
  egcm/model/text.py                    # TextTower
  egcm/model/vision.py                  # VisionTower (one bucket)
  egcm/model/embedder.py                # TextEmbedder / ImageEmbedder wrappers (pool + project + normalize)
  egcm/weights.py                       # safetensors → rewrite mapping
  egcm/convert.py                       # torch → mlpackage (+ multifunction merge)
  egcm/compress.py                      # int8 / palettize
  egcm/probe.py                         # activation max-abs probe
  egcm/fixtures.py                      # fixture manifest helpers
  stages/s0_reference/{README.md,run.py,build_fixtures.py}
  stages/s1_rewrite/{README.md,run.py}
  stages/s2_coreml_fp32/{README.md,run.py}
  stages/s3_coreml_ane/{README.md,run.py}
  stages/s4_compress/{README.md,run.py}
  stages/s5_swift/{README.md,run.py}
  stages/s6_device/{README.md,run.py}
  stages/s7_publish/{README.md,run.py}
  scripts/__init__.py  scripts/gen_status.py  scripts/gen_results.py  scripts/check_gates_adr.py  scripts/check_large_files.py
  tests/conftest.py  tests/test_report.py  tests/test_gates.py  tests/test_provenance.py  tests/test_stage.py
  tests/test_scripts.py  tests/test_metrics.py  tests/test_buckets.py  tests/test_norm.py
  tests/test_rope.py  tests/test_weights_map.py  tests/test_gates_adr.py
  fixtures/images/*.{jpg,png}  fixtures/images.jsonl  fixtures/texts.jsonl  fixtures/MANIFEST.sha256
  docs/decisions/ADR-0001-ane-rewrite.md  docs/decisions/ADR-0002-image-buckets.md  docs/decisions/ADR-0003-initial-gate-set.md  docs/RESULTS.md
  swift/Package.swift
  swift/Sources/EmbeddingGemma/{EmbeddingGemma.swift,EmbeddingTask.swift,Errors.swift,DeviceTier.swift,
       Buckets.swift,ImagePreprocessor.swift,TextEncoder.swift,ModelStore.swift,Checksums.swift}
  swift/Sources/EmbeddingIndex/{VectorIndex.swift,IndexFile.swift}
  swift/Sources/egcm-parity/main.swift  # s5 CLI
  swift/Tests/EmbeddingGemmaTests/*.swift  swift/Tests/EmbeddingIndexTests/*.swift
  Examples/PhotoSearchDemo/{project.yml, Sources/*.swift}
```

---

### Task 1: Repo scaffold, agent docs, toolchain

> **Implemented, then amended after code review.** The code blocks below are the original draft; the files in the repo are authoritative. Changes: torch pinned to 2.7.0 and coremltools to 9.0 (coremltools 9.0 sets `_TORCH_MAX_VERSION = "2.7.0"`); `gates.toml` tables are keyed by full stage name with the missing keys added (ADR-0003); the `stage` recipe is strict and runs under `uv run --exact --group <g>`; scripts run as `python -m scripts.<name>`; ruff is configured; `vendor/reference/SOURCE.txt` records the transformers commit.

**Files:** create `pyproject.toml`, `.gitignore`, `justfile`, `gates.toml`, `AGENTS.md`, `CLAUDE.md`, `LICENSE`, `NOTICE`, `vendor/reference/*`, `egcm/__init__.py`, `egcm/paths.py`.

- [ ] **Step 1: Install tools.** Run `mise use -g just@latest` (or `brew install just`), then `uv --version` and `just --version`.
- [ ] **Step 2: Write `pyproject.toml`.**

```toml
[project]
name = "egcm"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["numpy>=2", "pydantic>=2.7", "safetensors>=0.4", "pillow>=11", "tomli-w>=1"]

[dependency-groups]
convert = ["torch", "coremltools>=9", "pytest>=8"]
reference = ["torch", "transformers==5.20.2", "sentence-transformers>=6.1", "torchvision", "pytest>=8"]
dev = ["pytest>=8", "ruff>=0.6"]

[tool.uv]
conflicts = [[{ group = "convert" }, { group = "reference" }]]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Pin torch for `convert`.** Check the newest coremltools release notes for the officially supported torch version: `uv pip index versions coremltools`, then read `https://github.com/apple/coremltools/releases`. Pin `torch==<that version>` in the `convert` group. Run `uv sync --group convert`, then `uv run --group convert python -c "import torch,coremltools as ct;print(torch.__version__,ct.__version__)"`. Expected: prints both versions with no warning about an untested torch.
- [ ] **Step 4: Write `.gitignore`.**

```
artifacts/
*.mlpackage
*.mlmodelc
.venv/
__pycache__/
.build/
*.safetensors
Examples/PhotoSearchDemo/*.xcodeproj
.DS_Store
```

- [ ] **Step 5: Write `gates.toml`** (values from the spec §5).

```toml
# Every threshold lives here. Changing this file needs a new docs/decisions/ADR-*.md in the same commit.
#
# Each table is keyed by the full stage name. The comparison direction comes from the key:
#   min_*  -> metric >= value
#   max_*  -> metric <= value
#   bool   -> metric == value
# Any other key raises when the file is loaded. The prefix is the direction only; each metric's
# docstring (egcm/metrics.py) or stage README defines how it is aggregated.
# A "targets" subtable holds recorded-only numbers; it is never gated.

[s0_reference]
license_apache = true
key_inventory_written = true
deterministic = true

[s1_rewrite]
strict_load = true
min_cosine_text = 0.9999
min_cosine_image = 0.9999

[s2_coreml_fp32]
min_cosine_text = 0.9999
min_cosine_image = 0.9999

[s3_coreml_ane]
min_cosine_text = 0.995
min_cosine_image = 0.995
min_ane_fraction = 0.95

[s4_compress]
# min_cosine is the minimum over text and image vectors combined.
min_cosine = 0.99
min_recall_at_10 = 0.95
# 5th percentile over the 120 app queries (see egcm.metrics.spearman_topk).
min_p5_spearman_top50 = 0.9
max_download_mb = 300

[s5_swift]
token_ids_identical = true
# 1/255 rounded up.
max_pixel_abs_diff = 0.003922
# Minimum over text and image vectors combined.
min_cosine = 0.99

[s6_device]
max_peak_rss_mb = 800
min_devices = 1

[s6_device.targets]
target_ms_per_image_70 = 60

[s7_publish]
confirmed = true
```

- [ ] **Step 6: Write `egcm/paths.py`.**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIXTURES = ROOT / "fixtures"
GATES = ROOT / "gates.toml"
HF_REPO = "google/embeddinggemma-2"

def stage_dir(stage: str) -> Path:
    d = ARTIFACTS / stage
    d.mkdir(parents=True, exist_ok=True)
    return d
```

- [ ] **Step 7: Write the `justfile`.**

```just
set shell := ["bash", "-euo", "pipefail", "-c"]
stages := "s0_reference s1_rewrite s2_coreml_fp32 s3_coreml_ane s4_compress s5_swift s6_device s7_publish"

# Sync the convert environment and, once swift/ exists, resolve the Swift package.
# The reference environment is not synced here: `just stage 0` installs it on first use (uv run --exact).
setup:
    uv sync --group convert
    if [ -d swift ]; then (cd swift && swift package resolve); fi

status:
    uv run python -m scripts.gen_status

# Run stage N in the environment named by line 1 of its run.py (`# group: convert` or `# group: reference`).
# --exact makes the environment match that group alone, so reference-only packages (transformers) never
# leak into convert stages. STATUS.md is always regenerated, and the stage's exit code is kept.
stage n:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{n}}" in ''|*[!0-9]*) echo "error: stage must be a number, got '{{n}}'" >&2; exit 2;; esac
    name=$(printf '%s\n' {{stages}} | grep -x 's{{n}}_.*' || true)
    if [ -z "$name" ]; then
        echo "error: no stage numbered {{n}}. Stages: {{stages}}" >&2
        exit 2
    fi
    run="stages/$name/run.py"
    if [ ! -f "$run" ]; then
        echo "error: $run does not exist yet" >&2
        exit 2
    fi
    group=$(head -1 "$run" | sed -n 's/^# group: *//p')
    if [ -z "$group" ]; then
        echo "error: line 1 of $run must be '# group: convert' or '# group: reference'" >&2
        exit 2
    fi
    rc=0
    uv run --exact --group "$group" python -m "stages.$name.run" || rc=$?
    uv run --no-sync python -m scripts.gen_status >/dev/null || echo "warning: could not regenerate STATUS.md" >&2
    exit "$rc"

all:
    for n in 0 1 2 3 4 5; do just stage $n; done

test:
    uv run --exact --group convert pytest -q
    if [ -d swift ]; then (cd swift && swift test); fi

lint:
    uv run --group dev ruff check .

bench-device:
    just stage 6

publish *flags:
    uv run --exact --group convert python -m stages.s7_publish.run {{flags}}
```

- [ ] **Step 8: Write `AGENTS.md`** with the 7 rules from spec §4 verbatim, the command list, the stage table (spec §5), and a "Where things are" map (the file map above). Write `CLAUDE.md` containing only `See AGENTS.md.`
- [ ] **Step 9: Copy the reference sources.** Download the three upstream files into `vendor/reference/`, from the `v5.20.2` tag (the version the `reference` group installs); record the tag and its commit SHA (`gh api repos/huggingface/transformers/commits/v5.20.2 --jq .sha`) in `vendor/reference/SOURCE.txt`. Add `LICENSE` (Apache-2.0 full text) and a `NOTICE` that credits Google (model) and Hugging Face (reference code).
- [ ] **Step 10: Commit.** `git add -A && git commit -m "chore: scaffold repo, gates, agent docs"`

### Task 2: Report schema and gate loading

> **Implemented, then amended after code review.** The code blocks below are the original draft; the files in the repo are authoritative. Changes: `evaluate(report, stage_key)` lives in `egcm/gates.py` and loads its own gates (no `ops` argument; operator comes from the `min_`/`max_` prefix or a bool, anything else raises); `Report` is schema and atomic I/O only, forbids extra fields, has `status`, `git_dirty`, `code_hash`, and writes non-finite floats as strings; `egcm/provenance.py` and `egcm/stage.py` (`run_stage`) are new. **Every stage writes its report with `run_stage`, never with `Report(...)`, `evaluate` or `write_report` directly.**

**Files:** create `egcm/report.py`, `egcm/gates.py`, `tests/test_report.py`.

- [ ] **Step 1: Write the failing test.**

```python
# tests/test_report.py
import json
from egcm.report import Report, write_report, read_report, evaluate

def test_roundtrip_and_pass(tmp_path):
    r = Report(stage="s1_rewrite", git_sha="abc", inputs_sha256="x", metrics={"min_cosine_text": 0.99995},
               gate_values={"min_cosine_text": 0.9999})
    r = evaluate(r, {"min_cosine_text": ">="})
    assert r.passed and r.failures == []
    p = write_report(r, tmp_path)
    assert read_report(p).passed

def test_fail_records_reason(tmp_path):
    r = evaluate(Report(stage="s3_coreml_ane", git_sha="a", inputs_sha256="x",
                        metrics={"min_ane_fraction": 0.8}, gate_values={"min_ane_fraction": 0.95}),
                 {"min_ane_fraction": ">="})
    assert not r.passed
    assert "min_ane_fraction" in r.failures[0]

def test_max_gate(tmp_path):
    r = evaluate(Report(stage="s4_compress", git_sha="a", inputs_sha256="x",
                        metrics={"max_download_mb": 310.0}, gate_values={"max_download_mb": 300}),
                 {"max_download_mb": "<="})
    assert not r.passed
```

- [ ] **Step 2: Run it.** `uv run --group convert pytest tests/test_report.py -q`. Expected: FAIL with an ImportError.
- [ ] **Step 3: Implement.**

```python
# egcm/report.py
from __future__ import annotations
import json, subprocess, datetime as dt
from pathlib import Path
from pydantic import BaseModel, Field

class Report(BaseModel):
    stage: str
    git_sha: str
    inputs_sha256: str
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)
    gate_values: dict[str, float | int | bool] = Field(default_factory=dict)
    details: dict = Field(default_factory=dict)
    passed: bool = False
    failures: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
    finished_at: str | None = None

def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

def evaluate(r: Report, ops: dict[str, str]) -> Report:
    failures = []
    for key, gate in r.gate_values.items():
        if key not in r.metrics:
            failures.append(f"{key}: metric missing")
            continue
        val, op = r.metrics[key], ops.get(key, ">=")
        ok = {">=": val >= gate, "<=": val <= gate, "==": val == gate}[op]
        if not ok:
            failures.append(f"{key}: {val} not {op} {gate}")
    r.failures, r.passed = failures, not failures
    r.finished_at = dt.datetime.now(dt.UTC).isoformat()
    return r

def write_report(r: Report, directory: Path) -> Path:
    p = Path(directory) / "report.json"
    p.write_text(r.model_dump_json(indent=2))
    return p

def read_report(p: Path) -> Report:
    return Report.model_validate_json(Path(p).read_text())
```

```python
# egcm/gates.py
import tomllib
from egcm.paths import GATES

def gates(stage_key: str) -> dict:
    return tomllib.loads(GATES.read_text())[stage_key]

def ops_for(g: dict) -> dict[str, str]:
    return {k: ("<=" if k.startswith("max_") else "==" if isinstance(v, bool) else ">=") for k, v in g.items()}
```

- [ ] **Step 4: Run it.** Same command. Expected: 3 passed.
- [ ] **Step 5: Commit.** `git commit -am "feat: report schema and gate evaluation"`

### Task 3: Metrics

> **Implemented, then amended after code review.** The code blocks below are the original draft; the files in the repo are authoritative. Changes: `recall_at_k` and `spearman_topk` take keyword-only `ref=(q, d), new=(q, d)` and raise `ValueError` if k exceeds the number of docs; `spearman_topk` ranks the reference top-k inside the new model's full stable ranking (intruders count) and returns the 5th percentile over queries (gate `min_p5_spearman_top50`); `recall_at_k` is the mean. Tests use hand-computed values.

**Files:** create `egcm/metrics.py`, `tests/test_metrics.py`.

- [ ] **Step 1: Write the failing test.**

```python
import numpy as np
from egcm.metrics import cosine_rows, recall_at_k, spearman_topk

def test_cosine_identity():
    a = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    assert np.allclose(cosine_rows(a, a), 1.0, atol=1e-6)

def test_recall_perfect_and_partial():
    rng = np.random.default_rng(1)
    q, d = rng.normal(size=(4, 16)), rng.normal(size=(50, 16))
    assert recall_at_k(q, d, q, d, k=10) == 1.0
    assert recall_at_k(q, d, q, rng.normal(size=(50, 16)), k=10) < 0.6

def test_spearman_identical():
    rng = np.random.default_rng(2)
    q, d = rng.normal(size=(3, 16)), rng.normal(size=(60, 16))
    assert spearman_topk(q, d, q, d, k=50) > 0.999
```

- [ ] **Step 2: Run it.** `uv run --group convert pytest tests/test_metrics.py -q`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement.**

```python
# egcm/metrics.py
import numpy as np

def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def cosine_rows(a, b) -> np.ndarray:
    return np.sum(_norm(a) * _norm(b), axis=-1)

def _topk(q, d, k):
    s = _norm(q) @ _norm(d).T
    return np.argsort(-s, axis=1)[:, :k], s

def recall_at_k(q_ref, d_ref, q_new, d_new, k=10) -> float:
    ref, _ = _topk(q_ref, d_ref, k)
    new, _ = _topk(q_new, d_new, k)
    return float(np.mean([len(set(r) & set(n)) / k for r, n in zip(ref, new)]))

def spearman_topk(q_ref, d_ref, q_new, d_new, k=50) -> float:
    ref, _ = _topk(q_ref, d_ref, k)
    _, s_new = _topk(q_new, d_new, k)
    out = []
    for i, idx in enumerate(ref):
        new_rank = np.argsort(np.argsort(-s_new[i, idx]))
        out.append(1 - 6 * np.sum((np.arange(k) - new_rank) ** 2) / (k * (k * k - 1)))
    return float(np.min(out))
```

- [ ] **Step 4: Run it.** Expected: 3 passed.
- [ ] **Step 5: Commit.** `git commit -am "feat: parity metrics"`

### Task 4: Buckets and resize math

> **Implemented, then amended after code review.** The code blocks below are the original draft; the files in the repo are authoritative. Changes: ten buckets (tall and wide added), validated inputs, and tests that compare the port with the vendored upstream function.

**Files:** create `egcm/buckets.py`, `tests/test_buckets.py`, `docs/decisions/ADR-0002-image-buckets.md`.

- [ ] **Step 1: Write the failing test.**

```python
from egcm.buckets import BUCKETS, choose_bucket, aspect_preserving_size

def test_buckets_are_fixed_points():
    for b in BUCKETS:
        assert aspect_preserving_size(b.height, b.width, budget=b.budget) == (b.height, b.width)
        assert (b.height // 48) * (b.width // 48) == b.tokens

def test_choose():  # signature is (height, width, budget)
    assert choose_bucket(3024, 4032, 70).name == "landscape"   # w/h = 1.33
    assert choose_bucket(4032, 3024, 70).name == "portrait"
    assert choose_bucket(1000, 1000, 280).name == "square"
    assert choose_bucket(2532, 1170, 70).name == "tall"        # iPhone screenshot, 1170 wide x 2532 tall
    assert choose_bucket(1080, 1920, 70).name == "wide"        # 16:9

def test_matches_upstream_formula():
    # f = sqrt(630*256 / (3024*4032)) = 0.11501 -> ideal 347.8 x 463.7 -> floor to multiples of 48 = 336 x 432
    assert aspect_preserving_size(3024, 4032, budget=70) == (336, 432)
```

- [ ] **Step 2: Run it.** Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** (port of `get_aspect_ratio_preserving_size` from `vendor/reference/image_processing_gemma4.py`).

```python
# egcm/buckets.py
import math
from dataclasses import dataclass

PATCH, POOL = 16, 3
SIDE = PATCH * POOL  # 48

@dataclass(frozen=True)
class Bucket:
    budget: int; name: str; height: int; width: int
    @property
    def tokens(self) -> int: return (self.height // SIDE) * (self.width // SIDE)
    @property
    def grid(self) -> tuple[int, int]: return self.height // PATCH, self.width // PATCH
    @property
    def function_name(self) -> str: return f"embed_image_{self.budget}_{self.name}"

BUCKETS = (
    Bucket(70, "portrait", 480, 336), Bucket(70, "landscape", 336, 480), Bucket(70, "square", 384, 384),
    Bucket(280, "portrait", 960, 672), Bucket(280, "landscape", 672, 960), Bucket(280, "square", 768, 768),
)

def aspect_preserving_size(height: int, width: int, budget: int) -> tuple[int, int]:
    max_patches = budget * POOL * POOL
    target_px = max_patches * PATCH * PATCH
    f = math.sqrt(target_px / (height * width))
    th = int(math.floor(f * height / SIDE)) * SIDE
    tw = int(math.floor(f * width / SIDE)) * SIDE
    max_side = (max_patches // POOL**2) * SIDE
    if th == 0 and tw == 0:
        raise ValueError("image too small")
    if th == 0:
        th, tw = SIDE, min(int(math.floor(width / height)) * SIDE, max_side)
    elif tw == 0:
        tw, th = SIDE, min(int(math.floor(height / width)) * SIDE, max_side)
    return th, tw

def choose_bucket(height: int, width: int, budget: int) -> Bucket:
    r = width / height
    name = "portrait" if r < 0.83 else "landscape" if r > 1.2 else "square"
    return next(b for b in BUCKETS if b.budget == budget and b.name == name)
```

(This draft code has three buckets per budget. The repo version has ten, the 0.6 / 0.83 / 1.2 / 1.67 thresholds above, and input validation.)

- [ ] **Step 4: Run it.** Expected: pass. If `test_matches_upstream_formula` fails, recompute the expected value by hand from the formula; the test value must come from the formula, not from a guess. (The draft's `(288, 384)` was wrong, and the draft's `test_choose` passed width and height in the wrong order.)
- [ ] **Step 5: Write ADR-0002.** Context: the upstream resize keeps aspect ratio, which gives variable shapes, and Core ML on the Neural Engine needs fixed shapes. Decision: ten fixed-point buckets (table above). Consequence: up to about 21% aspect stretch in the middle three buckets and none bounded in tall and wide (see the ADR table), measured by the s4 recall gate. Then commit: `git commit -am "feat: image buckets"`.

### Task 5: Fixtures (s0, part a)

> **Implemented, with these changes.** At JPEG quality 92 the 100 Commons photos alone came to 53 MB at a 1600-px max side, so they are saved at quality 85 (fixtures/ is 55 MB). Memes, blurry and dark images are made at a 1200-px max side. The chosen Commons files are pinned in `fixtures/commons_sources.json` (rebuilds reuse it; `--requery` searches again) and at most two files share a two-word title prefix, because search returned bursts of the same scene. Search uses `generator=search` with `incategory:CC-Zero` plus a license check on `extmetadata.LicenseShortName`. Fonts are macOS system fonts (Helvetica, Impact, Courier New); each synthetic row records its font. The rendering code is split into `stages/s0_reference/commons.py` and `synth.py`. A rebuild on this machine gives byte-identical files.

**Files:** create `stages/s0_reference/build_fixtures.py`, `egcm/fixtures.py`, `fixtures/texts.jsonl`, `fixtures/images.jsonl`, `fixtures/images/*`, `fixtures/MANIFEST.sha256`.

- [ ] **Step 1: Write `fixtures/texts.jsonl`.** 200 lines, `{"id": "t000", "text": "...", "kind": "query"|"document"}`:
  - 120 app queries, varied: "screenshots of tweets", "blurry photos", "receipts", "memes", "a photo of a dog", "whiteboard notes", "food I ate", "photos of documents", "nude or private photos", "duplicate selfies", "parking ticket", "boarding pass", "a sunset at the beach", … (write all 120, no generator);
  - 79 documents (one-sentence captions);
  - 1 long string of 128 or more tokens (a paragraph).
- [ ] **Step 2: Write `build_fixtures.py`.** It must be deterministic (seeded) and produce ~200 images:
  - **100 CC0 photos** from Wikimedia Commons: the API `https://commons.wikimedia.org/w/api.php` with `generator=categorymembers` on `Category:CC-Zero` image subcategories, filtering `extmetadata.LicenseShortName in {"CC0","Public domain"}`. Categories: people, dogs, cats, food, landscapes, buildings, documents, cars. Store the `source_url`, `license`, `author` per file.
  - **30 synthetic screenshots** rendered with PIL at 1170×2532: chat bubbles, tweet cards, settings lists, using DejaVu fonts. Seeded random text from `texts.jsonl` documents.
  - **20 synthetic memes:** a CC0 photo plus Impact-style top and bottom caption.
  - **20 blurry** (Gaussian radius 6–12 of CC0 photos), **10 dark** (brightness × 0.15), **10 near-duplicates** (crop 5% + JPEG q=60 of CC0 photos), **10 receipts/documents** (PIL-rendered receipts at 1000×2400).
  - Every image is saved as JPEG q=92 (PNG for screenshots).
  - `fixtures/images.jsonl` rows: `{"id","file","category","source_url","license","author","height","width"}`.
  - Finally, write `MANIFEST.sha256` (`sha256  path` lines, sorted).
- [ ] **Step 3: Run it.** `uv run --exact --group reference python -m stages.s0_reference.build_fixtures`. Expected: about 200 files, the manifest written, and every Commons row has a license in {CC0, Public domain}.
- [ ] **Step 4: Write `egcm/fixtures.py`** with `load_texts()`, `load_images()` (returns rows plus a PIL loader), and `verify_manifest() -> list[str]` (the mismatches). Add `tests/test_fixtures_manifest.py`, which asserts `verify_manifest() == []`.
- [ ] **Step 5: Commit.** `git add fixtures stages/s0_reference/build_fixtures.py egcm/fixtures.py tests/test_fixtures_manifest.py && git commit -m "feat: CC0 fixtures"`. The total fixture size must stay under 60 MB. If it doesn't, downscale the Commons photos to a 1600-px max side at build time.

### Task 6: s0 reference run

> **Implemented, with these changes.** Token ids come from `model.preprocess` (`model.tokenize` is deprecated in sentence-transformers 6.x); the official embedding path is `model.encode({"image": img}, processing_kwargs={"image": {"max_soft_tokens": budget}})`, which agreed with the manual path at cosine ≥ 0.9999998. Model helpers are in `stages/s0_reference/reference.py`. `images_<budget>.npz` stores the bucket image as uint8 HxWx3 (deflated, written one array at a time) after asserting the bit-exact pixel contract; `egcm.fixtures.load_bucket_pixels` turns it back into model input. After review: per-stage code hashing (`STAGE_CODE`), fixtures converted to sRGB, pinned sha256 per Commons download, and the model's sha256 checked. `keys.json` and `image_token_layout.json` matched the Facts and the Task 12 mapping table; the recorded image and text sequences were added to the Facts. The long text t199 was trimmed from 156 to 128 tokens so that it fits `embed_text_128`.

**Files:** create `stages/s0_reference/run.py`, `stages/s0_reference/README.md`.

- [ ] **Step 1: Write `run.py`** (first line `# group: reference`). It must:
  1. Use the pinned revision `egcm.paths.HF_REVISION` (never look up `main`). Download into `artifacts/hf/` with `snapshot_download(HF_REPO, revision=HF_REVISION)` and write `artifacts/s0_reference/revision.txt`.
  2. Write the key inventory: open `model.safetensors` with `safetensors.safe_open` and dump `{key: [shape, dtype]}` to `keys.json`.
  3. Load `SentenceTransformer(path, model_kwargs={"torch_dtype": torch.float32}, config_kwargs={"audio_config": None})` on CPU, in `eval()` mode.
  4. **Texts:** for every fixture text, with the `query` prompt for kind=query and the `document` prompt otherwise, record `input_ids` (via `model.tokenize`), and `embedding = model.encode(text, prompt_name=..., normalize_embeddings=True)` at the full 768 dims. Save `texts.npz` (`ids_<id>` int32 arrays, `emb` [N,768] float32, `order` ids).
  5. **Images:** for each image and each budget in {70, 280}:
     - `raw`: the reference processor on the original image (`processor(images=img, text="<|image|>", images_kwargs={"max_soft_tokens": budget})`), then the model forward, then sentence-transformers mean pooling and normalize. Record `emb_raw`.
     - `bucket`: resize the original to the bucket size with torchvision `resize(..., interpolation=BICUBIC, antialias=True)` on the uint8 tensor (exactly what the processor does), then run the processor (a fixed point, so it does not resize again) and the model. Record `emb_bucket`, `pixel_values` (float32 patches), `image_position_ids`, `input_ids`.
     - Use the processor's own embedding path. To get the embedding: `model.encode({"image": img}, ...)` if sentence-transformers supports that dict input (check `encode`'s signature), otherwise call the inner HF model and mean-pool `last_hidden_state` over `attention_mask` manually, then normalize. Assert both paths agree on 2 images (cosine > 0.99999) and record which one was used.
     - Save `images_<budget>.npz`.
  6. **Debug intermediates** for 2 fixtures (`t000`, plus the first image at budget 70): text `inputs_embeds`, `per_layer_inputs`, hidden after layers 0, 5 and 23, and the final pre-pool; vision patch-embed output, encoder layer 0 and 15, pooler output, `embed_vision` output. Use forward hooks. Save `debug.npz`.
  7. **Token layout:** save `image_token_layout.json` with the full `input_ids` for one image per bucket, mapped to token strings. It **must** show where `<|image>`, the placeholders, `<image|>` and any bos/eos sit.
  8. **Determinism:** run steps 4–5 twice on the first 10 items and assert bitwise equality → `metrics.deterministic = True`.
  9. Wrap steps 1–8 in `fill(report)` and call `run_stage("s0_reference", fill)` (`egcm.stage`). `fill` sets `metrics` `deterministic`, `license_apache` (True when the model card license is `apache-2.0`) and `key_inventory_written`, and `details = {"revision", "license": "apache-2.0", "embedding_path"}`. Provenance, `inputs_sha256` (manifest + revision), evaluation against `[s0_reference]` and the exit code are `run_stage`'s job.
- [ ] **Step 2: Run it.** `just stage 0`. Expected: report passes. `image_token_layout.json` shows the sequence. `keys.json` has `vision_tower.*`, `embed_vision.*` and `language_model.*` keys.
- [ ] **Step 3: Write `stages/s0_reference/README.md`.** Input, outputs (the list above), gate, and known failure modes: "transformers changed: the version is pinned in pyproject (`transformers==5.20.2`); if you bump it, re-vendor `vendor/reference/` from the same tag and diff the model code"; "sentence-transformers `encode` dict input changed: use the manual pooling path".
- [ ] **Step 4: Update the plan facts.** If `image_token_layout.json` or `keys.json` contradict the facts at the top of this plan, update the "Facts" section in this file, in the same commit, before Task 7.
- [ ] **Step 5: Commit.** `git commit -am "feat(s0): reference vectors and debug intermediates"` (artifacts stay gitignored).

### Task 7: Config and the scale-safe RMSNorm

> **Implemented, then amended after review.** The code below is the original draft; `egcm/config.py` and `egcm/model/norm.py` are authoritative. Changes: `eps/s²` is floored at `2**-24` (`eps_eff`) so fp16 never sees eps = 0; `config.load()` resolves `per_layer_config` per layer and rejects configs the rewrite does not implement (non-default rope, `standardize`, clipped linears, `attention_bias`, `sliding_window < buckets.MAX_SEQ`).

**Files:** create `egcm/config.py`, `egcm/model/norm.py`, `tests/test_norm.py`.

- [ ] **Step 1: Write the failing test.**

```python
import torch
from egcm.model.norm import RMSNormCF

def ref_rms(x, w, eps=1e-6):  # x: (S, C) upstream layout
    return x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5) * (w if w is not None else 1)

def test_matches_upstream_layout():
    x = torch.randn(7, 512) * 30
    w = torch.rand(512) + 0.5
    n = RMSNormCF(512, with_scale=True); n.weight.data = w.clone()
    y = n(x.T.reshape(1, 512, 1, 7))
    assert torch.allclose(y.reshape(512, 7).T, ref_rms(x, w), atol=1e-5)

def test_scale_safe_identical_in_fp32():
    x = torch.randn(1, 512, 1, 9) * 4000
    a, b = RMSNormCF(512, with_scale=False), RMSNormCF(512, with_scale=False, prescale=256.0)
    assert torch.allclose(a(x), b(x), rtol=1e-5, atol=1e-6)

def test_scale_safe_no_overflow_fp16():
    x = (torch.randn(1, 512, 1, 9) * 4000).half()
    y = RMSNormCF(512, with_scale=False, prescale=256.0).half()(x)
    assert torch.isfinite(y).all()
```

- [ ] **Step 2: Run it.** Expected: FAIL (ImportError).
- [ ] **Step 3: Implement.**

```python
# egcm/model/norm.py
import torch
from torch import nn

class RMSNormCF(nn.Module):
    """RMSNorm over dim 1 of (B, C, 1, S). `prescale` keeps x² inside fp16 range: result is identical in exact math."""
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True, prescale: float = 1.0):
        super().__init__()
        self.eps, self.prescale, self.with_scale = eps, prescale, with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.prescale
        xs = x / s if s != 1.0 else x
        y = xs * torch.rsqrt(xs.pow(2).mean(1, keepdim=True) + self.eps / (s * s))
        if self.with_scale:
            y = y * self.weight.view(1, -1, 1, 1)
        return y
```

```python
# egcm/config.py
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class TextCfg:
    hidden: int; layers: int; heads: int; ffn: int; ple_dim: int; out_dim: int; vocab: int; eps: float
    layer_types: tuple[str, ...]; head_dim: dict; kv_heads: dict; theta: dict; pad_id: int

@dataclass(frozen=True)
class VisionCfg:
    hidden: int; layers: int; heads: int; head_dim: int; ffn: int; patch: int; pool: int; theta: float; eps: float; pos_size: int

def load(hf_dir: Path) -> tuple[TextCfg, VisionCfg, dict]:
    c = json.loads((Path(hf_dir) / "config.json").read_text())
    t, v = c["text_config"], c["vision_config"]
    hd = {"sliding_attention": t["head_dim"], "full_attention": t["per_layer_config"]["05"]["head_dim"]}
    kv = {"sliding_attention": t["num_key_value_heads"], "full_attention": t["per_layer_config"]["05"]["num_key_value_heads"]}
    th = {k: t["rope_parameters"][k]["rope_theta"] for k in ("sliding_attention", "full_attention")}
    text = TextCfg(t["hidden_size"], t["num_hidden_layers"], t["num_attention_heads"], t["intermediate_size"],
                   t["hidden_size_per_layer_input"], t["embedding_dim"], t["vocab_size"], t["rms_norm_eps"],
                   tuple(t["layer_types"]), hd, kv, th, t["pad_token_id"])
    vis = VisionCfg(v["hidden_size"], v["num_hidden_layers"], v["num_attention_heads"], v["head_dim"],
                    v["intermediate_size"], v["patch_size"], v["pooling_kernel_size"],
                    v["rope_parameters"]["rope_theta"], v["rms_norm_eps"], v["position_embedding_size"])
    return text, vis, c
```

Note: `per_layer_config` is keyed by layer index strings (`"05"`, `"11"`, …) in `config.json`. Assert at load that every full-attention layer listed in `layer_types` has the same per-layer config.

- [ ] **Step 4: Run it.** Expected: 3 passed.
- [ ] **Step 5: Commit.** `git commit -am "feat: config + channels-first scale-safe RMSNorm"`

### Task 8: RoPE tables

**Files:** create `egcm/model/rope.py`, `tests/test_rope.py`.

- [ ] **Step 1: Write the failing test.** It compares against the upstream formulas, re-implemented inline in the test from `vendor/reference` (1-D: `rotate_half` with `cat(freqs, freqs)`; 2-D: the x half then the y half, each `rotate_half` within 32 channels).

```python
import torch
from egcm.model.rope import rope_1d, rope_2d, apply_rope_cf

def ref_rotate_half(x):
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-b, a), -1)

def test_1d_matches_upstream():
    S, D, theta = 11, 256, 1e4
    inv = 1.0 / (theta ** (torch.arange(0, D, 2).float() / D))
    f = torch.arange(S)[:, None].float() * inv
    cos, sin = torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()
    q = torch.randn(4, S, D)
    ref = q * cos + ref_rotate_half(q) * sin
    c, s = rope_1d(S, D, theta)                      # (1, D, 1, S) each
    out = apply_rope_cf(q.permute(0, 2, 1).unsqueeze(2), c, s, halves=1)   # (4, D, 1, S)
    assert torch.allclose(out.squeeze(2).permute(0, 2, 1), ref, atol=1e-5)

def test_2d_matches_upstream():
    Hp, Wp, D, theta = 6, 9, 64, 100.0
    xs = torch.arange(Wp).repeat(Hp); ys = torch.arange(Hp).repeat_interleave(Wp)
    inv = 1.0 / (theta ** (torch.arange(0, D // 2, 2).float() / (D // 2)))
    fx, fy = xs[:, None].float() * inv, ys[:, None].float() * inv
    cos = torch.cat([fx.cos(), fx.cos(), fy.cos(), fy.cos()], -1)
    sin = torch.cat([fx.sin(), fx.sin(), fy.sin(), fy.sin()], -1)
    q = torch.randn(12, Hp * Wp, D)
    halves = []
    for k in range(2):
        sl = slice(32 * k, 32 * (k + 1))
        halves.append(q[..., sl] * cos[:, sl] + ref_rotate_half(q[..., sl]) * sin[:, sl])
    ref = torch.cat(halves, -1)
    c, s = rope_2d(Hp, Wp, D, theta)
    out = apply_rope_cf(q.permute(0, 2, 1).unsqueeze(2), c, s, halves=2)
    assert torch.allclose(out.squeeze(2).permute(0, 2, 1), ref, atol=1e-5)
```

- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement.**

```python
# egcm/model/rope.py
import torch

def rope_1d(S: int, D: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, D, 2).float() / D))
    f = torch.arange(S)[:, None].float() * inv
    e = torch.cat([f, f], -1)                              # (S, D)
    return e.cos().T.reshape(1, D, 1, S), e.sin().T.reshape(1, D, 1, S)

def rope_2d(Hp: int, Wp: int, D: int, theta: float):
    xs = torch.arange(Wp).repeat(Hp).float(); ys = torch.arange(Hp).repeat_interleave(Wp).float()
    inv = 1.0 / (theta ** (torch.arange(0, D // 2, 2).float() / (D // 2)))
    fx, fy = xs[:, None] * inv, ys[:, None] * inv
    cos = torch.cat([fx.cos(), fx.cos(), fy.cos(), fy.cos()], -1)
    sin = torch.cat([fx.sin(), fx.sin(), fy.sin(), fy.sin()], -1)
    S = Hp * Wp
    return cos.T.reshape(1, D, 1, S), sin.T.reshape(1, D, 1, S)

def _rotate_half_cf(x):  # rotate along dim 1
    h = x.shape[1] // 2
    return torch.cat((-x[:, h:], x[:, :h]), 1)

def apply_rope_cf(x, cos, sin, halves: int):
    """x: (B, D, 1, S). halves=1 → standard; halves=2 → axial (x-half, y-half)."""
    if halves == 1:
        return x * cos + _rotate_half_cf(x) * sin
    n = x.shape[1] // halves
    parts = [x[:, i*n:(i+1)*n] * cos[:, i*n:(i+1)*n] + _rotate_half_cf(x[:, i*n:(i+1)*n]) * sin[:, i*n:(i+1)*n]
             for i in range(halves)]
    return torch.cat(parts, 1)
```

- [ ] **Step 4: Run it.** Expected: 2 passed.
- [ ] **Step 5: Commit.** `git commit -am "feat: rope tables (1d + axial 2d)"`

### Task 9: ANE attention

> **Implemented, then amended after review.** The code below is the original draft; `egcm/model/attention.py`, `egcm/model/masks.py`, `tests/test_attention.py` and `tests/upstream.py` are authoritative. The main change: the attention mask is additive and **key-major, `(1, S, 1, 1)`**, built by `egcm.model.masks.key_mask(valid)` from a `(1, S)` 0/1 tensor (or `None` for no padding). `AttentionCF` adds it to the `(1, S_k, 1, S_q)` logits directly, with no per-head transpose.

**Files:** create `egcm/model/attention.py`, `tests/test_attention.py`.

- [ ] **Step 1: Write the failing test.** It checks against a direct `(S, H, D)` implementation of the upstream `eager_attention_forward` with `scaling=1.0`, GQA `repeat_kv`, q/k RMSNorm with weights, v RMSNorm without weight, RoPE after the norm, and an additive mask (0 or −1e4) on padded keys.

```python
import torch
from egcm.model.attention import AttentionCF
from egcm.model.rope import rope_1d

def ref_attn(x, m, mask):  # x: (S, C)
    S = x.shape[0]
    q = (x @ m.q.weight.squeeze(-1).squeeze(-1).T).view(S, m.heads, m.hd)
    k = (x @ m.k.weight.squeeze(-1).squeeze(-1).T).view(S, m.kv, m.hd)
    v = (x @ m.v.weight.squeeze(-1).squeeze(-1).T).view(S, m.kv, m.hd)
    rms = lambda t, w: t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * (w if w is not None else 1)
    q, k, v = rms(q, m.q_norm.weight), rms(k, m.k_norm.weight), rms(v, None)
    inv = 1.0 / (1e4 ** (torch.arange(0, m.hd, 2).float() / m.hd))
    f = torch.arange(S)[:, None].float() * inv; e = torch.cat([f, f], -1)
    rot = lambda t: torch.cat((-t[..., m.hd // 2:], t[..., : m.hd // 2]), -1)
    q = q * e.cos()[:, None] + rot(q) * e.sin()[:, None]
    k = k * e.cos()[:, None] + rot(k) * e.sin()[:, None]
    rep = m.heads // m.kv
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    w = torch.einsum("qhd,khd->hqk", q, k) + mask[None, None, :]
    o = torch.einsum("hqk,khd->qhd", w.softmax(-1), v).reshape(S, -1)
    return o @ m.o.weight.squeeze(-1).squeeze(-1).T

def test_matches_reference_with_padding():
    torch.manual_seed(0)
    m = AttentionCF(hidden=512, heads=4, kv_heads=2, head_dim=256, eps=1e-6, rope_halves=1)
    for p in m.parameters(): p.data.normal_(0, 0.05)
    S = 9; x = torch.randn(S, 512)
    mask = torch.zeros(S); mask[7:] = -1e4
    cos, sin = rope_1d(S, 256, 1e4)
    out = m(x.T.reshape(1, 512, 1, S), cos, sin, mask.view(1, 1, 1, S))
    ref = ref_attn(x, m, mask)
    assert torch.allclose(out.reshape(512, S).T[:7], ref[:7], atol=1e-4)
```

- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement** (1×1 convs, per-head einsum, mask added over keys).

```python
# egcm/model/attention.py
import torch
from torch import nn
from egcm.model.norm import RMSNormCF
from egcm.model.rope import apply_rope_cf

def conv1x1(cin, cout):
    return nn.Conv2d(cin, cout, 1, bias=False)

class AttentionCF(nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, eps, rope_halves):
        super().__init__()
        self.heads, self.kv, self.hd, self.halves = heads, kv_heads, head_dim, rope_halves
        self.q, self.k = conv1x1(hidden, heads * head_dim), conv1x1(hidden, kv_heads * head_dim)
        self.v, self.o = conv1x1(hidden, kv_heads * head_dim), conv1x1(heads * head_dim, hidden)
        self.q_norm = RMSNormCF(head_dim, eps)
        self.k_norm = RMSNormCF(head_dim, eps)
        self.v_norm = RMSNormCF(head_dim, eps, with_scale=False)

    def _split(self, t, n):  # (1, n*hd, 1, S) -> list of (1, hd, 1, S)
        return t.split(self.hd, dim=1)

    def forward(self, x, cos, sin, mask):
        qs = [apply_rope_cf(self.q_norm(h), cos, sin, self.halves) for h in self._split(self.q(x), self.heads)]
        ks = [apply_rope_cf(self.k_norm(h), cos, sin, self.halves) for h in self._split(self.k(x), self.kv)]
        vs = [self.v_norm(h) for h in self._split(self.v(x), self.kv)]
        rep = self.heads // self.kv
        outs = []
        for i, q in enumerate(qs):
            k, v = ks[i // rep], vs[i // rep]
            w = torch.einsum("bchq,bkhc->bkhq", q, k.transpose(1, 3))  # (1, S_k, 1, S_q)
            w = (w + mask.transpose(1, 3)).softmax(dim=1)
            outs.append(torch.einsum("bkhq,bchk->bchq", w, v))
        return self.o(torch.cat(outs, 1))
```

- [ ] **Step 4: Run it.** Expected: PASS. If the einsum index letters are off, fix the einsum strings; never change the reference.
- [ ] **Step 5: Commit.** `git commit -am "feat: ANE-layout attention with GQA and qkv norms"`

### Task 10: Text tower and embedder

**Files:** create `egcm/model/text.py`, part of `egcm/model/embedder.py`.

- [ ] **Step 1: Implement `TextTower`** (verified in Task 12 against s0 `debug.npz` and the vectors; no separate synthetic test).

```python
# egcm/model/text.py
import torch
from torch import nn
from egcm.config import TextCfg
from egcm.model.norm import RMSNormCF
from egcm.model.attention import AttentionCF, conv1x1
from egcm.model.rope import rope_1d

GELU = nn.GELU(approximate="tanh")

class MLP(nn.Module):
    def __init__(self, c, f):
        super().__init__(); self.gate, self.up, self.down = conv1x1(c, f), conv1x1(c, f), conv1x1(f, c)
    def forward(self, x): return self.down(GELU(self.gate(x)) * self.up(x))

class Layer(nn.Module):
    def __init__(self, cfg: TextCfg, kind: str):
        super().__init__()
        c, e = cfg.hidden, cfg.eps
        self.kind = kind
        self.attn = AttentionCF(c, cfg.heads, cfg.kv_heads[kind], cfg.head_dim[kind], e, rope_halves=1)
        self.mlp = MLP(c, cfg.ffn)
        self.in_norm, self.post_attn = RMSNormCF(c, e), RMSNormCF(c, e)
        self.pre_ff, self.post_ff = RMSNormCF(c, e), RMSNormCF(c, e)
        self.ple_gate, self.ple_proj = conv1x1(c, cfg.ple_dim), conv1x1(cfg.ple_dim, c)
        self.post_ple = RMSNormCF(c, e)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, h, ple_i, rope, mask):
        cos, sin = rope[self.kind]
        h = h + self.post_attn(self.attn(self.in_norm(h), cos, sin, mask))
        h = h + self.post_ff(self.mlp(self.pre_ff(h)))
        h = h + self.post_ple(self.ple_proj(GELU(self.ple_gate(h)) * ple_i))
        return h * self.layer_scalar

class TextTower(nn.Module):
    """Input: inputs_embeds (1, C, 1, S) already scaled by sqrt(C); additive key-major mask (1, S, 1, 1) from key_mask."""
    def __init__(self, cfg: TextCfg, seq_len: int):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([Layer(cfg, k) for k in cfg.layer_types])
        self.ple_proj = conv1x1(cfg.hidden, cfg.layers * cfg.ple_dim)
        self.ple_norm = RMSNormCF(cfg.ple_dim, cfg.eps)
        self.norm = RMSNormCF(cfg.hidden, cfg.eps)
        self.out_proj = conv1x1(cfg.hidden, cfg.out_dim)
        for kind in ("sliding_attention", "full_attention"):
            c, s = rope_1d(seq_len, cfg.head_dim[kind], cfg.theta[kind])
            self.register_buffer(f"cos_{kind}", c); self.register_buffer(f"sin_{kind}", s)

    def forward(self, emb, mask):
        cfg = self.cfg
        ple = self.ple_proj(emb) * cfg.hidden ** -0.5                       # (1, L*P, 1, S)
        ples = [self.ple_norm(p) for p in ple.split(cfg.ple_dim, dim=1)]   # L x (1, P, 1, S)
        rope = {k: (getattr(self, f"cos_{k}"), getattr(self, f"sin_{k}")) for k in ("sliding_attention", "full_attention")}
        h = emb
        for i, layer in enumerate(self.layers):
            h = layer(h, ples[i], rope, mask)
        return self.out_proj(self.norm(h))                                  # (1, 768, 1, S)
```

- [ ] **Step 2: Implement `egcm/model/embedder.py`.**

```python
# egcm/model/embedder.py
import torch
from torch import nn
from egcm.model.masks import key_mask

def masked_mean_l2(tokens, valid):  # tokens (1, D, 1, S); valid (1,1,1,S) in {0,1}
    pooled = (tokens * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True)
    v = pooled.reshape(1, -1)
    return v * torch.rsqrt((v * v).sum(-1, keepdim=True))

class TextEmbedder(nn.Module):
    """Core ML entry point for text. `table` is the scaled embedding table; gather happens here."""
    def __init__(self, tower, table: torch.Tensor):
        super().__init__(); self.tower = tower
        self.register_buffer("table", table)   # (V, C) already * sqrt(C)
    def forward(self, input_ids, attention_mask):  # (1, S) int32, (1, S) float 0/1
        emb = self.table[input_ids[0].long()].T.unsqueeze(0).unsqueeze(2)   # (1, C, 1, S)
        valid = attention_mask.view(1, 1, 1, -1)                            # pooling weights over S
        mask = key_mask(attention_mask)                                     # (1, S, 1, 1) additive
        return masked_mean_l2(self.tower(emb, mask), valid)
```

- [ ] **Step 3: Commit.** `git commit -am "feat: text tower + embedder"`

### Task 11: Vision tower and image embedder

**Files:** create `egcm/model/vision.py`, append to `egcm/model/embedder.py`.

- [ ] **Step 1: Implement.**

```python
# egcm/model/vision.py
import torch
from torch import nn
import torch.nn.functional as F
from egcm.config import VisionCfg
from egcm.buckets import Bucket
from egcm.model.norm import RMSNormCF
from egcm.model.attention import AttentionCF, conv1x1
from egcm.model.masks import key_mask
from egcm.model.rope import rope_2d
from egcm.model.text import MLP

class VLayer(nn.Module):
    def __init__(self, cfg: VisionCfg):
        super().__init__()
        c, e = cfg.hidden, cfg.eps
        self.attn = AttentionCF(c, cfg.heads, cfg.heads, cfg.head_dim, e, rope_halves=2)
        self.mlp = MLP(c, cfg.ffn)
        self.in_norm, self.post_attn = RMSNormCF(c, e), RMSNormCF(c, e)
        self.pre_ff, self.post_ff = RMSNormCF(c, e), RMSNormCF(c, e)
    def forward(self, h, cos, sin, mask):
        h = h + self.post_attn(self.attn(self.in_norm(h), cos, sin, mask))
        return h + self.post_ff(self.mlp(self.pre_ff(h)))

class VisionTower(nn.Module):
    """pixels (1,3,H,W) in [0,1] → soft tokens (1, 512, 1, T) in text space (after embed_vision)."""
    def __init__(self, cfg: VisionCfg, bucket: Bucket, text_hidden: int):
        super().__init__()
        self.cfg, self.bucket = cfg, bucket
        Hp, Wp = bucket.grid
        self.patch = nn.Conv2d(3, cfg.hidden, cfg.patch, stride=cfg.patch, bias=True)
        self.register_buffer("pos", torch.zeros(1, cfg.hidden, Hp, Wp))       # filled by weights.py
        c, s = rope_2d(Hp, Wp, cfg.head_dim, cfg.theta)
        self.register_buffer("cos", c); self.register_buffer("sin", s)
        self.register_buffer("mask", key_mask(torch.ones(1, Hp * Wp)))        # (1, S, 1, 1), all 0: no padding in a bucket (None is equivalent)
        self.layers = nn.ModuleList([VLayer(cfg) for _ in range(cfg.layers)])
        self.mm_norm = RMSNormCF(cfg.hidden, cfg.eps, with_scale=False)       # sqrt(768) pooler scale dropped: scale-invariant
        self.mm_proj = conv1x1(cfg.hidden, text_hidden)

    def forward(self, pixels):
        Hp, Wp = self.bucket.grid
        x = self.patch(pixels) + self.pos                                     # (1, C, Hp, Wp)
        h = x.reshape(1, self.cfg.hidden, 1, Hp * Wp)
        for layer in self.layers:
            h = layer(h, self.cos, self.sin, self.mask)
        g = F.avg_pool2d(h.reshape(1, self.cfg.hidden, Hp, Wp), self.cfg.pool, self.cfg.pool)
        t = g.reshape(1, self.cfg.hidden, 1, -1)                              # row-major blocks == upstream kernel_idx order
        return self.mm_proj(self.mm_norm(t))
```

```python
# append to egcm/model/embedder.py
class ImageEmbedder(nn.Module):
    """Builds the full token sequence: prefix text embeds + soft tokens + suffix text embeds (constants from s0 layout)."""
    def __init__(self, vision, tower, prefix_emb: torch.Tensor, suffix_emb: torch.Tensor):
        super().__init__(); self.vision, self.tower = vision, tower
        self.register_buffer("prefix", prefix_emb)   # (1, C, 1, P)
        self.register_buffer("suffix", suffix_emb)   # (1, C, 1, Q)
    def forward(self, pixels):
        soft = self.vision(pixels)
        emb = torch.cat([self.prefix, soft, self.suffix], dim=-1)
        S = emb.shape[-1]
        valid = torch.ones(1, 1, 1, S)
        return masked_mean_l2(self.tower(emb, key_mask(torch.ones(1, S))), valid)
```

The text tower's RoPE buffers are sized to the sequence length, so each image bucket gets its own `TextTower(seq_len = P + tokens + Q)`. The weights are shared through the multifunction merge in Task 14.

- [ ] **Step 2: Commit.** `git commit -am "feat: vision tower + image embedder"`

### Task 12: Weight mapping and s1 parity

**Files:** create `egcm/weights.py`, `tests/test_weights_map.py`, `stages/s1_rewrite/run.py`, `stages/s1_rewrite/README.md`.

- [ ] **Step 1: Write the failing test** (uses `artifacts/s0_reference/keys.json`; skipped when it's absent).

```python
import json, pytest
from egcm.paths import ARTIFACTS
from egcm.weights import mapping_for_keys

KEYS = ARTIFACTS / "s0_reference" / "keys.json"

@pytest.mark.skipif(not KEYS.exists(), reason="run `just stage 0` first")
def test_every_needed_key_mapped_and_no_audio():
    keys = json.loads(KEYS.read_text())
    used, unused = mapping_for_keys(keys)
    assert all(k.split(".")[0] in ("audio_tower", "embed_audio") or "audio" in k for k in unused), unused[:10]
    assert any(k.startswith("language_model.layers.23.") for k in used)
    assert any("vision_tower.encoder.layers.15." in k for k in used)
```

- [ ] **Step 2: Implement `egcm/weights.py`.**
  - `load_hf(dir) -> dict[str, Tensor]` (FP32).
  - `mapping_for_keys(keys) -> (used, unused)`.
  - `build_text_tower(cfg, sd, seq_len) -> TextTower`.
  - `build_vision_tower(vcfg, sd, bucket, text_hidden) -> VisionTower`.
  - `text_table(sd) -> Tensor` (embed weight × hidden**0.5).

  The mapping is explicit, one line per parameter family, with linear weights reshaped `(out,in) → (out,in,1,1)`. The actual prefixes come from `keys.json`. Expected shape:

  ```
  language_model.embed_tokens.weight                         -> table (× sqrt(512))
  language_model.ple.per_layer_model_projection.weight       -> ple_proj
  language_model.ple.per_layer_projection_norm.weight        -> ple_norm.weight
  language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight  -> layers.i.attn.{q,k,v,o}
  language_model.layers.{i}.self_attn.{q,k}_norm.weight      -> layers.i.attn.{q,k}_norm.weight
  language_model.layers.{i}.input_layernorm.weight           -> layers.i.in_norm.weight
  language_model.layers.{i}.post_attention_layernorm.weight  -> layers.i.post_attn.weight
  language_model.layers.{i}.pre_feedforward_layernorm.weight -> layers.i.pre_ff.weight
  language_model.layers.{i}.post_feedforward_layernorm.weight-> layers.i.post_ff.weight
  language_model.layers.{i}.mlp.{gate,up,down}_proj.weight   -> layers.i.mlp.{gate,up,down}
  language_model.layers.{i}.ple_block.per_layer_input_gate.weight -> layers.i.ple_gate
  language_model.layers.{i}.ple_block.per_layer_projection.weight -> layers.i.ple_proj
  language_model.layers.{i}.ple_block.post_per_layer_input_norm.weight -> layers.i.post_ple.weight
  language_model.layers.{i}.layer_scalar                     -> layers.i.layer_scalar
  language_model.norm.weight                                 -> norm.weight
  language_model.embedding_projection.weight                 -> out_proj
  vision_tower.patch_embedder.input_proj.weight  W(768,768)  -> patch.weight = W.view(768,16,16,3).permute(0,3,1,2)*2 ; patch.bias = -W.sum(1)
  vision_tower.patch_embedder.position_embedding_table (2,10240,768) -> pos[:, :, y, x] = T[0][x] + T[1][y]
  vision_tower.encoder.layers.{i}.self_attn.{q,k,v,o}_proj.linear.weight -> layers.i.attn.{q,k,v,o}
  vision_tower.encoder.layers.{i}.self_attn.{q,k}_norm.weight            -> layers.i.attn.{q,k}_norm.weight
  vision_tower.encoder.layers.{i}.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight
  vision_tower.encoder.layers.{i}.mlp.{gate,up,down}_proj.linear.weight
  embed_vision.embedding_projection.weight                   -> mm_proj
  ```

  If a key carries a `model.` prefix, strip it. If the key names differ from this list, `keys.json` is the authority: update this table in the plan and the code together. Load with `load_state_dict(strict=True)` on our module, so a missing or extra key fails loudly. Keys that exist in the HF weights but aren't used must all be audio keys (the test above).
- [ ] **Step 3: Write `stages/s1_rewrite/run.py`** (`# group: convert`). It must:
  1. Build the text tower per text length bucket (16/32/64/128) and pad the ids from `texts.npz` to the smallest bucket that fits.
  2. Compute the embedding and compare it with `emb` → `min_cosine_text`.
  3. For each image and each budget, build `ImageEmbedder` for the chosen bucket, with `prefix_emb`/`suffix_emb` taken from `image_token_layout.json` (embed the non-placeholder ids with the table).
  4. Run on `pixels` reconstructed from s0's `pixel_values`. Unpatchify with the inverse of `convert_image_to_patches` and check it reproduces s0's patches exactly, as a sanity check.
  5. Compare with `emb_bucket` → `min_cosine_image`.
  6. Also compare `debug.npz` intermediates and record per-tensor max abs diff in `details`. These are diagnostic, not gated.
  7. Put these in `fill(report)` and call `run_stage("s1_rewrite", fill)`; set `metrics` `strict_load` (True when `load_state_dict(strict=True)` succeeded), `min_cosine_text` and `min_cosine_image`.
- [ ] **Step 4: Run it.** `just stage 1`. Expected: pass at ≥ 0.9999. If it fails, the debug intermediates show the first diverging tensor; fix that module. The README lists the likely suspects: the patch flatten order, the axial RoPE half order, a missing layer_scalar, mask sign, or the PLE reshape order (layer-major vs dim-major: `reshape(S, L, P)` means layer-major, so `split(P, dim=1)` is correct).
- [ ] **Step 5: Commit.** `git commit -am "feat(s1): weight mapping + rewrite parity"` and update STATUS.md (`just status`).

### Task 13: s2 Core ML FP32 conversion

**Files:** create `egcm/convert.py`, `stages/s2_coreml_fp32/run.py`, its README, and `docs/decisions/ADR-0001-ane-rewrite.md`.

- [ ] **Step 1: Implement `egcm/convert.py`.**

```python
import coremltools as ct, numpy as np, torch
from pathlib import Path

TEXT_LENGTHS = (16, 32, 64, 128)

def convert_module(module, example_inputs, input_specs, out_path: Path, precision):
    module.eval()
    traced = torch.jit.trace(module, example_inputs, strict=False)
    m = ct.convert(traced, inputs=input_specs, outputs=[ct.TensorType(name="embedding")],
                   compute_precision=precision, minimum_deployment_target=ct.target.iOS18,
                   convert_to="mlprogram")
    m.save(str(out_path))
    return m

def text_specs(L):
    return [ct.TensorType("input_ids", shape=(1, L), dtype=np.int32),
            ct.TensorType("attention_mask", shape=(1, L), dtype=np.float32)]

def image_specs(h, w):
    return [ct.TensorType("pixels", shape=(1, 3, h, w), dtype=np.float32)]

def merge_multifunction(parts: dict[str, Path], out_path: Path, default: str):
    desc = ct.utils.MultiFunctionDescriptor()
    for fn, p in parts.items():
        desc.add_function(str(p), src_function_name="main", target_function_name=fn)
    desc.default_function_name = default
    ct.utils.save_multifunction(desc, str(out_path))
```

- [ ] **Step 2: Write `stages/s2_coreml_fp32/run.py`.** Convert each text length and each of the 10 image buckets at `ct.precision.FLOAT32` into `artifacts/s2_coreml_fp32/parts/<fn>.mlpackage`. Merge into `EmbeddingGemma2.mlpackage`. Load each function with `ct.models.MLModel(path, function_name=fn, compute_units=ct.ComputeUnit.CPU_ONLY)`, predict all fixtures, compare with s0 → `min_cosine_text` / `min_cosine_image`. Record the package size in MB. Report via `run_stage("s2_coreml_fp32", fill)`.
- [ ] **Step 3: Run it.** `just stage 2`. Expected: pass. Known failure modes (README):
  - `einsum` pattern unsupported → rewrite as matmul with explicit transposes.
  - The int64 gather on `table` → cast ids to int32 (already int32); if the gather still fails, use `torch.index_select`.
  - Tracing the bool-free mask → fine.
- [ ] **Step 4: Write ADR-0001** (why the ANE rewrite, citing Apple's ml-ane-transformers) and commit: `git commit -am "feat(s2): fp32 core ml multifunction conversion"`.

### Task 14: s3 FP16 on the Neural Engine and the probe

**Files:** create `egcm/probe.py`, `stages/s3_coreml_ane/run.py`, README.

- [ ] **Step 1: Implement `probe.py`.** Register forward hooks on every `RMSNormCF`, `AttentionCF`, `MLP` and residual output of the s1 FP32 rewrite (residual outputs captured by wrapping `Layer.forward`). Record the max |x| per module over all fixtures. Write `artifacts/s3_coreml_ane/probe.json` sorted descending. Flag > 30000 as `overflow_risk`, and > 3000 for squared terms inside norms (`x²` above 65504 means |x| > 256 inside the mean, so in practice every norm whose *input* max exceeds 256 needs a prescale).
- [ ] **Step 2: Apply prescales automatically.** For each `RMSNormCF` whose input max exceeds 256, set `prescale = 2**ceil(log2(max/64))`. This is exact in math. Persist the chosen prescales in `artifacts/s3_coreml_ane/prescales.json` and log them in the report details.
- [ ] **Step 3: Write `run.py`.** Re-convert all functions with `compute_precision=ct.precision.FLOAT16`. Load with `compute_units=ct.ComputeUnit.CPU_AND_NE`. Measure:
  - cosines against s0;
  - **Neural Engine fraction**: `ct.models.compute_plan.MLComputePlan.load_from_path(compiled_path, compute_units=CPU_AND_NE)`, then iterate the program's operations, get `get_compute_device_usage_for_mlprogram_operation(op).preferred_compute_device`, and count `MLNeuralEngineComputeDevice` over all ops that have a usage (the minimum over functions).
  - Compile first with `ct.utils.compile_model`.
  - Report via `run_stage("s3_coreml_ane", fill)`, with `min_cosine_text`, `min_cosine_image`, `min_ane_fraction`.
- [ ] **Step 4: Run it.** `just stage 3`. If cosine fails:
  1. Read probe.json and check that the residual stream itself (not just norm inputs) exceeds 30000. If it does, apply mitigation (b): scale the embeddings and every residual add by `1/s`. RMSNorm after it is invariant, `layer_scalar` is unchanged, and the final `norm` is invariant, so this is exact math. Implement it as a `residual_scale` constructor arg on `TextTower`/`Layer` that divides `emb` and nothing else. Since every sub-block output is re-normalized by post-norms (whose weights then produce full-scale values), also divide each post-norm weight by `s`.
  2. If (b) still isn't enough, use mitigation (c): `op_selector` keeps the flagged ops FP32 (`ct.transform.FP16ComputePrecision(op_selector=...)`), then re-measure the Neural Engine fraction.
  3. If it still fails at ≥ 95%, take the **GPU fallback**: ADR-0004 (ADR-0003 is the initial gate set), convert with `compute_units=CPU_AND_GPU` gating, and change `gates.toml` `[s3_coreml_ane] min_ane_fraction` → `0` **with ADR-0004 in the same commit**.
- [ ] **Step 5: Commit** the passing state: `git commit -am "feat(s3): fp16 ANE conversion with probe-driven prescales"`.

### Task 15: s4 compression

**Files:** create `egcm/compress.py`, `stages/s4_compress/run.py`, README.

- [ ] **Step 1: Implement.**

```python
import coremltools as ct
import coremltools.optimize.coreml as cto

def int8(m):
    cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", granularity="per_channel"))
    return cto.linear_quantize_weights(m, cfg)

def palettize4(m):
    cfg = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(nbits=4, mode="kmeans", granularity="per_grouped_channel", group_size=16))
    return cto.palettize_weights(m, cfg)
```

- [ ] **Step 2: Write `run.py`.** Start from the s3 parts. Produce the variants `int8` and `pal4`, and the mixed `pal4` + embedding table at int8 (an `op_name_configs` override for the gather `const`). Then for each variant:
  1. merge the multifunction package;
  2. compute cosines, recall@10 and the 5th-percentile Spearman of the top 50, with `q` = the 120 query texts (`kind=query` only; the 80 document-style texts are not used for ranking) and `d` = the bucketed images at budget 70 and, separately, at 280. Both must pass. Call `recall_at_k(ref=(q_ref, d_ref), new=(q, d), k=10)` and `spearman_topk(ref=..., new=..., k=50)`; the metrics are `min_recall_at_10` and `min_p5_spearman_top50`, and `min_cosine` is the minimum over text and image vectors combined. Use the reference `q_ref`/`d_ref` from s0 **raw** (exact-aspect) image vectors, so bucketing distortion is included;
  3. measure the download size as the zipped `.mlpackage` (`shutil.make_archive`).

  Pick the smallest variant that passes everything. Report the winning variant's metrics through `run_stage("s4_compress", fill)`. Write `artifacts/s4_compress/EmbeddingGemma2.mlpackage` plus `checksums.json` (sha256 per file inside the zip and of the zip itself). Record every variant's metrics in details.
- [ ] **Step 3: Run it.** `just stage 4`. If recall fails only because of bucketing (raw vs bucket recall at FP32 in s1 also below 0.95), go back to ADR-0002 and add buckets: an extra 3:4 portrait at 70 = 432×336 (9×7 = 63 tokens). Never relax the gate.
- [ ] **Step 4: Commit.** `git commit -am "feat(s4): compressed variants + selection"`

### Task 16: Swift package, part 1 (tokenizer, buckets, preprocessing)

**Files:** create `swift/Package.swift`, `Sources/EmbeddingGemma/{Buckets.swift,ImagePreprocessor.swift,TextEncoder.swift,EmbeddingTask.swift,Errors.swift}`, and tests.

- [ ] **Step 1: Write `Package.swift`.**

```swift
// swift-tools-version: 6.0
import PackageDescription
let package = Package(
  name: "EmbeddingGemma",
  platforms: [.iOS(.v18), .macOS(.v15)],
  products: [
    .library(name: "EmbeddingGemma", targets: ["EmbeddingGemma"]),
    .library(name: "EmbeddingIndex", targets: ["EmbeddingIndex"]),
  ],
  dependencies: [.package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0")],
  targets: [
    .target(name: "EmbeddingGemma", dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
            resources: [.copy("Resources/tokenizer")]),
    .target(name: "EmbeddingIndex"),
    .executableTarget(name: "egcm-parity", dependencies: ["EmbeddingGemma"]),
    .testTarget(name: "EmbeddingGemmaTests", dependencies: ["EmbeddingGemma"]),
    .testTarget(name: "EmbeddingIndexTests", dependencies: ["EmbeddingIndex"]),
  ]
)
```

  Check the latest swift-transformers tag with `gh api repos/huggingface/swift-transformers/releases/latest --jq .tag_name` and use that as the `from:` version. Copy `tokenizer.json` and `tokenizer_config.json` from `artifacts/hf/` into `Sources/EmbeddingGemma/Resources/tokenizer/` (this is committed: about 30 MB. If it's over 20 MB, download it with the model instead and keep it out of git: `ModelStore` fetches it from the same HF revision).
- [ ] **Step 2: `Buckets.swift`** mirrors `egcm/buckets.py` exactly (same ten-bucket table, same thresholds 0.6 / 0.83 / 1.2 / 1.67 with the same inclusive and exclusive edges). Test `BucketsTests` with the same cases as `tests/test_buckets.py`.
- [ ] **Step 3: `ImagePreprocessor.swift`.**
  - Input `CGImage` → choose the bucket → resize to the bucket size with vImage `vImageScale_ARGB8888` and `kvImageHighQualityResampling` (Lanczos-like).
  - **The parity question:** the reference uses torchvision bicubic + antialias. The test compares Swift against s0's `pixel_values` for 20 images within `s5.max_pixel_abs_diff` (1/255 per pixel, 99th percentile). If vImage misses it, implement separable antialiased bicubic (a = −0.5, matching torch's `upsample_bicubic2d_aa`) with Accelerate on planar float buffers. That's the fallback, and `ImagePreprocessorTests` decides.
  - Output `MLMultiArray` `[1,3,H,W]` Float32 in 0–1. Draw into an sRGB `CGContext` first so orientation and colour space are normalized.
- [ ] **Step 4: `TextEncoder.swift`.** Load `AutoTokenizer.from(modelFolder:)` from the resource folder; `encode(text:prompt:) -> [Int32]`, prepending the prompt string. Pad to the smallest length in [16, 32, 64, 128] and truncate at 128. The mask is 1 for real tokens. Tests: for 50 fixture texts, the ids equal s0's ids exactly. The fixture ids are exported by a small script `stages/s5_swift/export_fixtures.py` to `swift/Tests/EmbeddingGemmaTests/Fixtures/text_ids.json` (committed, small).
- [ ] **Step 5: Run it.** `cd swift && swift test --filter "Buckets|TextEncoder|ImagePreprocessor"`. Expected: PASS.
- [ ] **Step 6: Commit.** `git commit -am "feat(swift): tokenizer, buckets, preprocessing parity"`

### Task 17: Swift package, part 2 (model runtime and ModelStore)

**Files:** create `EmbeddingGemma.swift`, `DeviceTier.swift`, `ModelStore.swift`, `Checksums.swift`, and tests.

- [ ] **Step 1: `DeviceTier.swift`.** `static func defaultBudget() -> Int` reads `sysctlbyname("hw.machine")`: iPhone16,1 and newer (A17 Pro+) → 280, otherwise 70. macOS → 280. Test it with an injected machine string.
- [ ] **Step 2: `ModelStore.swift`.**
  - `func ensureModel(repo:revision:progress:) async throws -> URL` downloads `https://huggingface.co/<repo>/resolve/<revision>/EmbeddingGemma2.mlpackage.zip` and `checksums.json` with `URLSession` download tasks (resumable via `resumeData`, stored on cancel or error).
  - It verifies the SHA-256 of the zip (CryptoKit), unzips with `Foundation`'s `FileManager` + `Archive`. There's no system unzip API on iOS, so to avoid a dependency **publish an uncompressed `.mlpackage` directory listing**: `files.json` = [path, size, sha256] for each file. ModelStore downloads each file and verifies each hash.
  - It compiles with `MLModel.compileModel(at:)` into `Application Support/EmbeddingGemma/<revision>/model.mlmodelc`, and keeps the previous revision folder until the new compile succeeds, then deletes it.
  - Tests use a local `file://` server folder built from s4 artifacts (skipped when they're absent) plus a tampered-hash case → `.checksumMismatch`.
- [ ] **Step 3: `EmbeddingGemma.swift`.** The actor from spec §8:
  - `load` → ModelStore → for each needed function, `MLModel(contentsOf:configuration:)` with `MLModelConfiguration.functionName = fn` and `computeUnits = .cpuAndNeuralEngine`. Image functions are loaded lazily per bucket at the chosen budget; text functions lazily per length.
  - `embed(image:)` → preprocess → predict → take the `embedding` (768) → truncate to `dimensions` → L2 normalize → `[Float]`.
  - `embed(query:)` uses the prompt `task: search result | query: `; `embed(text:task:)` maps `EmbeddingTask.document → "title: none | text: "`, `.classification → "task: classification | query: "`, `.clustering → "task: clustering | query: "`.
  - `modelRevision` is the HF commit SHA.
- [ ] **Step 4: Run it.** `swift test`. Expected: PASS (end-to-end tests skip cleanly if `artifacts/s4_compress` is missing, with message "run just stage 4").
- [ ] **Step 5: Commit.** `git commit -am "feat(swift): model runtime + verified downloader"`

### Task 18: EmbeddingIndex

**Files:** create `Sources/EmbeddingIndex/{VectorIndex.swift,IndexFile.swift}`, `Tests/EmbeddingIndexTests/VectorIndexTests.swift`.

- [ ] **Step 1: Write failing tests.**

```swift
import Testing
import Foundation
@testable import EmbeddingIndex

func rand(_ n: Int, _ d: Int, seed: UInt64) -> [[Float]] {
  var s = seed
  return (0..<n).map { _ in
    let v = (0..<d).map { _ -> Float in s = s &* 6364136223846793005 &+ 1; return Float(Int64(bitPattern: s >> 11) % 2000) / 1000 - 1 }
    let n = sqrt(v.reduce(0) { $0 + $1 * $1 }); return v.map { $0 / n }
  }
}

@Test func topKMatchesBruteForce() throws {
  let url = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
  let idx = try VectorIndex(url: url, dimensions: 64, modelRevision: "r1")
  let vs = rand(500, 64, seed: 7)
  for (i, v) in vs.enumerated() { try idx.upsert(id: "a\(i)", vector: v) }
  let q = vs[42]
  let hits = idx.search(q, k: 5)
  #expect(hits.first?.id == "a42")
  let brute = vs.enumerated().map { ($0.offset, zip($0.element, q).reduce(Float(0)) { $0 + $1.0 * $1.1 }) }
    .sorted { $0.1 > $1.1 }.prefix(5).map { "a\($0.0)" }
  #expect(hits.map(\.id) == Array(brute))
}

@Test func persistsAndRejectsOtherRevision() throws {
  let url = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
  let idx = try VectorIndex(url: url, dimensions: 8, modelRevision: "r1")
  try idx.upsert(id: "x", vector: [1,0,0,0,0,0,0,0]); try idx.save()
  let again = try VectorIndex(url: url, dimensions: 8, modelRevision: "r1")
  #expect(again.vector(for: "x")?.first == 1)
  #expect(throws: VectorIndexError.revisionMismatch(expected: "r2", found: "r1")) {
    _ = try VectorIndex(url: url, dimensions: 8, modelRevision: "r2")
  }
}

@Test func removeAndFilter() throws {
  let url = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
  let idx = try VectorIndex(url: url, dimensions: 4, modelRevision: "r")
  try idx.upsert(id: "a", vector: [1,0,0,0]); try idx.upsert(id: "b", vector: [0.9,0.1,0,0])
  try idx.remove(id: "a")
  #expect(idx.search([1,0,0,0], k: 2).map(\.id) == ["b"])
  #expect(idx.search([1,0,0,0], k: 2, filter: { $0 != "b" }).isEmpty)
}
```

- [ ] **Step 2: Run them.** `swift test --filter VectorIndex`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **Type:** `public final class VectorIndex` (not `Sendable`; one owner, normally an actor). A struct would silently copy 50+ MB on write.
  - **Storage:** one contiguous `[Float]` buffer `slotCount × dims` holding `Float(Float16(x))` (so values match after save/reload), `idBySlot: [String?]`, `slotByID: [String: Int]`, and a free-slot list for removals. `reserveCapacity(_:)` pre-sizes it.
  - **`search`:** `vDSP_mmul` over the Float32 buffer, then partial top-k with a min-heap of `min(k, count)`, skipping filtered and removed slots. Empty index or `k <= 0` returns `[]`; `k` may be `Int.max`.
  - **`IndexFile` (version 2):** little-endian header `"EGIX"` + version u32 + dims u32 + count u32 + revision length + revision utf8, padded to 16 bytes, then the FP16 block, then an ids JSON. Load widens FP16→F32 straight from the mapped bytes (`vImageConvert_Planar16FtoPlanarF`). Save streams chunks (`vImageConvert_PlanarFtoPlanar16F`) to a temp file and renames it into place (atomic).
  - `VectorIndexError: Equatable { case revisionMismatch(expected:found:), dimensionMismatch(expected:found:), invalidDimensions(Int), unsupportedVersion(UInt32), corrupt }`.
- [ ] **Step 4: Add a perf test.** 50k × 256 random, `search` k=50 is < 20 ms on macOS release (`swift test -c release --filter perf`). It's annotated `.timeLimit` and only enforced in release.
- [ ] **Step 5: Run and commit.** `git commit -am "feat(swift): EmbeddingIndex"`

### Task 19: s5 Swift parity stage

**Files:** create `Sources/egcm-parity/main.swift`, `stages/s5_swift/{run.py,export_fixtures.py,README.md}`.

- [ ] **Step 1: The `egcm-parity` CLI.** Arguments: `--model <s4 mlpackage dir> --fixtures <repo>/fixtures --out <json>`. It compiles the model, then for every fixture text computes token ids and an embedding (768, unnormalized-truncation-free), and for every image at both budgets computes pixels and an embedding. It writes JSON `{texts: {id: {ids, emb}}, images: {id_budget: {emb, pixel_sample}}}`. `pixel_sample` = 4096 deterministic pixel values: stride-sampled from the `[1,3,H,W]` tensor.
- [ ] **Step 2: `run.py`** (`# group: convert`). Run `swift run -c release egcm-parity ...`. Compare:
  - ids against s0 (exact) → `token_ids_identical` (bool, gate `==`);
  - pixel samples against the same stride samples of s0's unpatchified pixels → `max_pixel_abs_diff` as the 99th percentile;
  - embeddings against s0 → `min_cosine` (minimum over text and image vectors combined).

  `token_ids_identical = true` is already in `[s5_swift]` (ADR-0003). Report via `run_stage("s5_swift", fill)`.
- [ ] **Step 3: Run it.** `just stage 5`. Expected: pass.
- [ ] **Step 4: Commit.** `git commit -am "feat(s5): swift end-to-end parity"`

### Task 20: PhotoSearchDemo and the s6 device benchmark

**Files:** create `Examples/PhotoSearchDemo/project.yml`, `Sources/{App.swift,LibraryIndexer.swift,SearchView.swift,BenchmarkRunner.swift}`, `stages/s6_device/run.py`.

- [ ] **Step 1: `project.yml`** (XcodeGen). App target `PhotoSearchDemo`, iOS 18, bundle `com.delavalom.egcm.demo`. The package dependency is a local path `../../swift`. The `NSPhotoLibraryUsageDescription` string is "Search your photos on-device." `DEVELOPMENT_TEAM` is read from the `asc`-configured team: look it up with `asc` (team id) and write it to `Examples/PhotoSearchDemo/Team.xcconfig`, which is gitignored.
- [ ] **Step 2: The app.**
  - Permission → index the newest 1,000 images (thumbnail 960 px via `PHImageManager`) → `EmbeddingIndex` → search field → 3-column grid of results.
  - `BenchmarkRunner` (launch arg `-benchmark`) indexes 1,000 images at budget 70, then 200 at 280. It measures per-image ms (median, p90), query ms, peak RSS (`task_info` `phys_footprint` sampled every 100 ms), `ProcessInfo.thermalState` at the end, and the device model. It writes `Documents/s6_device.json` and shows "DONE".
- [ ] **Step 3: `stages/s6_device/run.py`** (`# group: convert`).
  1. `xcodegen` + `xcodebuild -scheme PhotoSearchDemo -destination 'id=<udid>' build`, where the UDID is the first `xcrun devicectl list devices` entry with `Reality == physical`. Fail with "connect an iPhone with Developer Mode" if there is none.
  2. `xcrun devicectl device install app`, then `xcrun devicectl device process launch --console <bundle> -benchmark`.
  3. Poll `xcrun devicectl device copy from --domain-type appDataContainer --domain-identifier <bundle> --source Documents/s6_device.json` every 15 s for up to 30 min.
  4. `fill(report)` for `run_stage("s6_device", fill)` sets `max_peak_rss_mb` (the maximum over devices) and `min_devices` (the number of devices with a result file), and puts the measured ms per image in `details` next to `egcm.gates.targets("s6_device")["target_ms_per_image_70"]`. The target is recorded, never gated.
  5. **The first run needs photos on the device**, which the founder's phone has. Photos permission must be granted once by hand on the device: the script prints "Tap Allow Full Access on the phone" and waits.
- [ ] **Step 4: Run it.** `just stage 6` with the iPhone connected. Expected: report written.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(s6): device benchmark"` and commit `artifacts/s6_device/report.json` copied to `docs/results/s6_<device>.json` (reports are small; RESULTS.md reads them).

### Task 21: STATUS, RESULTS, the gates-ADR check, CI

**Files:** extend `scripts/gen_status.py` (a minimal version exists: it lists reports with status, passed and stale, and writes STATUS.md); create `scripts/gen_results.py`, `scripts/check_gates_adr.py`, `scripts/check_large_files.py`, `tests/test_gates_adr.py`, `.github/workflows/ci.yml`.

- [ ] **Step 1: Write the failing test** for `check_gates_adr` (a pure function over a list of changed paths):

```python
from scripts.check_gates_adr import violates

def test_gates_change_needs_adr():
    assert violates(["gates.toml"]) is True
    assert violates(["gates.toml", "docs/decisions/ADR-0005-x.md"]) is False
    assert violates(["egcm/metrics.py"]) is False
```

- [ ] **Step 2: Implement the scripts.**
  - `violates(paths) = "gates.toml" in paths and not any(p.startswith("docs/decisions/ADR-") for p in paths)`. `main()` reads `git diff --name-only origin/main...HEAD` (CI) or `HEAD~1` locally.
  - `gen_status.py` already reads every `artifacts/s*/report.json` and marks staleness with `egcm.provenance.is_stale` (code hash or inputs hash changed). Extend it to also read `docs/results/*.json` and show key metrics in the table. It writes STATUS.md, prints it, and lists "Next: just stage N" for the first stage that is not passed and current.
  - `gen_results.py` writes `docs/RESULTS.md` from the reports, including every s4 variant.
  - `check_large_files.py` fails on any tracked file over 5 MB, except `fixtures/images/*` (each ≤ 2 MB is enforced instead).
- [ ] **Step 3: Write `ci.yml`.** On `pull_request` and push to main, run on `macos-latest`: checkout → `astral-sh/setup-uv` → `uv sync --group convert` → `uv run pytest -q` → `uv run python -m scripts.check_gates_adr` → `uv run python -m scripts.check_large_files` → `uv run --group dev ruff check .` → `uv run python -c "from egcm.fixtures import verify_manifest as v; assert v()==[]"` → `cd swift && swift build && swift test`.
- [ ] **Step 4: Run it.** `uv run --group convert pytest -q && just status`. Expected: PASS, and STATUS.md shows every stage.
- [ ] **Step 5: Commit.** `git commit -am "chore: status/results generation, gates-ADR CI check"`

### Task 22: s7 publish (guarded)

**Files:** create `stages/s7_publish/run.py`, README.

- [ ] **Step 1: Implement.**
  1. Refuse unless `--confirm` is given **and** s0–s6 reports have `status == "passed"` and are not stale per `egcm.provenance.is_stale`. Write its own report with `run_stage("s7_publish", fill)` with `metrics.confirmed = True` only when `--confirm` was given and the upstream check passed.
  2. Read `HF_REPO_TARGET` from the environment and exit with a message if it's unset.
  3. Create the HF model repo (public) via `huggingface_hub.create_repo(exist_ok=True)`.
  4. Upload the uncompressed `EmbeddingGemma2.mlpackage/` directory plus `files.json`, the tokenizer files, a README model card (base model, license Apache-2.0, parity table from RESULTS.md, usage via the Swift package) and `NOTICE`.
  5. Record the resulting commit SHA as `revision` and write `swift/Sources/EmbeddingGemma/DefaultModel.swift`: `public extension EmbeddingGemma { static let defaultRepo = "<target>"; static let defaultRevision = "<sha>" }`.
  6. `git tag v1.0.0`, `gh repo edit --visibility public --accept-visibility-change-consequences`, push the tags, and `gh release create v1.0.0 --notes-file docs/RESULTS.md`.
- [ ] **Step 2: Don't run it in this plan.** It runs after the founder approves with the final numbers. The HF token comes from the founder via a secure prompt (`request_secret`) or their logged-in `hf` CLI.
- [ ] **Step 3: Commit.** `git commit -am "feat(s7): guarded publish"`

---

## Self-review notes

- **Spec coverage:** every row in spec §5 maps to a task:
  - s0: Tasks 5–6; s1: 7–12; s2: 13; s3: 14; s4: 15; s5: 16–19; s6: 20; s7: 22.
  - The agent-first layout (spec §4) is Tasks 1 and 21; the Swift API (spec §8) is Tasks 16–18; the demo (spec §9) is Task 20; CI (spec §10) is Task 21.
- **Known open item resolved at runtime, with a defined procedure:** the exact HF key names and the image token layout. Task 6 records them; Task 12 treats `keys.json` as the authority, and Task 6 step 4 updates this plan when facts differ.
- **Type names are consistent across tasks:** `RMSNormCF`, `AttentionCF`, `TextTower(cfg, seq_len)`, `VisionTower(cfg, bucket, text_hidden)`, `TextEmbedder`, `ImageEmbedder`, `Bucket.function_name`, `VectorIndex`, `VectorIndexError`.
