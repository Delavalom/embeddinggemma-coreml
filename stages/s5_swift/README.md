# s5_swift

The Swift package on macOS: token ids and preprocessed pixels against s0, and its vectors against Python Core ML and
the FP reference on the same inputs (ADR-0008).

Run: `just stage 5` (the `convert` group). Needs fresh passing s0, s1 and s4. It runs on s4's two device archives,
one per image budget (ADR-0010; `details.budgets.<budget>.device_archive` in s4's report, sha256 from
`checksums.json`), each unpacked into `artifacts/s5_swift/unpacked/b<budget>/` (`egcm.manifest.unpack`): the package
and tokenizer bytes a phone downloads. Recorded as `details.s4_recipe` and `details.device_archives` (per budget:
`name`, `sha256`).

**Which package runs what** (`models.py`): egcm-parity and `runtime.py` get `--model 70=<pkg> --model 280=<pkg>`.
Image vectors of each budget come from that budget's package. Text vectors come from budget 70's package in both
(egcm-parity and Python Core ML; `details.text_budget`): both packages hold the same text functions, which s4 checks
bitwise against the 14-function package. A bare `--model <pkg>` still means one 14-function package for every budget
(`--measure`, the contract model). egcm-parity's `model_revisions` (per budget) is recorded as
`details.swift_model_revisions`.

## What it does

Three heavy steps, each in its own process, one at a time (16 GB Mac). Peak RSS of each is in `details.peak_rss`.

1. `run.py` writes the synthetic pattern (`export_fixtures.synthetic`) at `SYNTHETIC_SIZES` as PNGs into
   `artifacts/s5_swift/synthetic/`, with torch's uint8 antialiased bicubic of each as the reference.
2. **Swift.** `swift run -c release egcm-parity` (`swift/Sources/egcm-parity/`) goes through the path an app takes:
   it puts each budget's package and the tokenizer (s0's HF snapshot, else `artifacts/tokenizer/<revision>/`, sha256-checked by
   `egcm.published.tokenizer_source`) in a local published folder with `files.json`, installs it with `ModelStore`
   (verify, compile) and loads `EmbeddingGemma.load(.file(...), dimensions: 768)` once per budget, on
   `.cpuAndNeuralEngine`. For every fixture text (prompt by kind: `query` or `document`) it records the token ids and
   the vector. For every fixture image and each synthetic PNG at budgets 70 and 280 it records the bucket, the sha256
   of the uint8 bucket image and 8,192 pixel values (the `export_fixtures` sampling rule), and for fixture images
   the vector. With `--pixels artifacts/s5_swift/pixels/` it also writes each fixture's whole bucket image as
   `<id>_<budget>.png` (lossless). Output: `artifacts/s5_swift/parity.json`.
3. **Python Core ML** (`runtime.py`): the same packages, compiled by coremltools one at a time, on the same compute
   units, one function at a time. Inputs are Swift's own token ids and bucket images (read back and sha256-checked by
   `pixels.py`). Output: `runtime.npz`.
4. **FP reference on Swift's pixels** (`fp_reference.py`, one process per budget): the s1 rewrite in FP32 on the CPU,
   for the images whose Swift bucket image is not byte-identical to s0's (the JPEGs: ImageIO decodes them, PIL
   decoded s0's). For the byte-identical ones (the PNGs) the FP reference is s0's `emb_bucket`. Output: `fp_<budget>.npz`.
5. `compare.py` and `task_level.py` (pure functions; `tests/test_s5_compare.py`, `tests/test_s5_task_level.py`) turn
   that into the metrics.

## Metrics

Gated (`gates.toml [s5_swift]`, ADR-0008):

- `token_ids_identical`: every one of the 200 texts has exactly s0's ids (`ids_<id>` in `texts.npz`), none missing.
- `max_pixel_abs_diff`: the 99th percentile of `|swift - reference|` (0-1 units) over the pixel samples of the
  decode-identical inputs (ADR-0004): the PNG fixtures against s0's stored bucket pixels and the synthetic PNGs
  against torch. It is `inf` if any image lands in another bucket than the reference or an output is missing or
  unexpected. `details.pixels` also has the JPEG numbers (`jpeg_pixel_p99`, decoder included, recorded only), the
  maxima, and the bit-exact counts per group.
- `min_cosine_runtime`: Swift vs Python Core ML on identical inputs, minimum over all 600 vectors (200 texts, 400
  images). `details.runtime.by_group` splits it by text kind and by budget and format. It is NaN (a failure) if a
  vector is missing on the Python side.
- `min_cosine`, `min_recall_at_10`, `min_p5_spearman_top50`, `min_t2i_top1_in_top10`, `max_i2i_label_p10_drop`: s4's
  task-level metrics (`egcm.ranking.metrics`, the same keys and thresholds) on the Swift vectors, against the FP
  reference on Swift's pixels (texts: s0's `emb`). Details: `details.vs_fp_on_swift_pixels`.

Recorded only:

- `details.min_cosine_vs_s0` and `details.cosine`: every vector against s0's (`emb`, `emb_bucket`), by text kind and by
  budget and format, with the five worst ids of each. This was the gate before ADR-0008. On JPEGs it includes the
  decoder.
- `details.vs_s0`: the task-level metrics against s0's PIL-decoded reference.
- `details.decoder`: the FP reference on Swift's pixels against s0, which is what the decoder alone costs (no Swift,
  no compression).
- `details.fp_reference`: per budget, how many image references came from the FP32 rewrite vs s0's `emb_bucket`.

## Gate

`gates.toml [s5_swift]`: `token_ids_identical == true`, `max_pixel_abs_diff <= 0.003922`, `min_cosine_runtime >=
0.9999`, `min_cosine >= 0.99`, `min_recall_at_10 >= 0.95`, `min_p5_spearman_top50 >= 0.9`,
`min_t2i_top1_in_top10 >= 0.95`, `max_i2i_label_p10_drop <= 0.03`.

## Measuring a package without a report

```
uv run --exact --group convert python -m stages.s5_swift.run --measure <pkg> [--s0 <dir>] [--compute-units cpuOnly] [--limit N] [--skip-fp]
```

runs the same comparison on any one package with the spec section 7 I/O (all 14 functions) and prints the metrics and details as JSON
(files go to `artifacts/s5_swift/measure/`). `--skip-fp` leaves out the FP reference and the task-level metrics. The
ranking metrics need at least 50 images, so a `--limit` below 50 requires it. On the contract model (below), 2026-10-24, M1 Pro: ids identical for
200/200 texts; pixels bit-exact for 60/60 PNG and 20/20 synthetic cases, so `max_pixel_abs_diff` 0; JPEG p99 2/255,
max 31/255, 0/340 bit-exact. Its cosines mean nothing (it is not a model of anything); they only show the plumbing.

## Unit-test fixtures

`swift test` checks the tokenizer and the image preprocessor against small committed files in
`swift/Tests/EmbeddingGemmaTests/Fixtures/`, written by:

```
uv run --exact --group convert python -m stages.s5_swift.export_fixtures [--s0 <artifacts/s0_reference>]
```

- `text_ids.json`: s0's token ids for all 200 fixture texts (prompt, `<bos>` and `<eos>` included).
- `pixels.json`, `images`: 20 photos (4 per bucket shape; 2 of the tall ones are PNG screenshots) at both budgets:
  the bucket s0 chose, the sha256 of s0's full uint8 HxWx3 bucket image, and 8,192 values sampled from its uint8
  `[3, H, W]` tensor at indices `(k * 3*H*W) // 8192` (base64).
- `pixels.json`, `synthetic`: a procedural lossless pattern (reproduced exactly in Swift) at 10 sizes covering all
  ten buckets, resized by torch's uint8 antialiased bicubic (which reproduces s0 bit for bit), sampled the same way.

**The pixel gate is taken on decode-identical inputs** (`docs/decisions/ADR-0004-pixel-gate-inputs.md`): the
synthetic images and the PNG photos. There the 99th percentile of `|swift - reference|` (0-1 units) is gated at
`[s5_swift].max_pixel_abs_diff` (1/255), and in fact every case is bit-exact. JPEG photos decode differently in
ImageIO and PIL, so on them the number includes the decoder: it is printed and bounded at 3/255 for regressions only.

Measured 2026-10-23 on the M1 Pro:

| Resize | Inputs | p99 | max | bit-exact |
|---|---|---|---|---|
| vImage `vImageScale_ARGB8888`, high quality (tried first) | 20 photos x 2 budgets | 11/255 | 42/255 | 0/40 |
| torch uint8 AA bicubic port (`Resampler.swift`, Accelerate) | 10 synthetic x 2 budgets | 0 | 0 | 20/20 |
| same | 2 PNG photos x 2 budgets | 0 | 0 | 4/4 |
| same | 16 JPEG photos x 2 budgets (decoder included) | 2/255 | 31/255 | 0/32 |

- `embeddings.json`: s0's normalized vectors for 8 texts and 6 images (both budgets), the targets of
  `EndToEndTests` (cosine >= `[s5_swift].min_cosine`), which run once `artifacts/s4_compress/` has a package and
  are skipped with "run just stage 4" until then.

- `pixels.json`, `large_synthetic`: 9x to 18x downscales (12 MP, 48 MP, 5000x1200, 1080x7000), SHA-256 only;
  48 MP is checked by the release perf test (`swift test -c release --filter perfLargeImages`).
- `truncation_ids.json`: 10 texts of 400 to 2,800 tokens, truncated by the reference tokenizer
  (`uv run --exact --group reference python -m stages.s5_swift.export_truncation --s0 ...`); it first checks the
  reference tokenizer reproduces s0's ids for all 200 fixture texts.

**Regenerate on arm64 (Apple silicon).** torch has two uint8 antialiased bicubic kernels: an AVX2 one on x86 and the
generic separable one elsewhere. s0 ran on arm64, so its pixels come from the generic kernel, which is what
`Resampler.swift` ports bit for bit. Exporting on an x86 machine would produce AVX2 outputs that can differ.

Rerun the export whenever s0 is rerun; `tests/test_s5_export.py` checks the files are present, current (same
`HF_REVISION`), cover every bucket, and stay under 2 MB.

The tokenizer (`tokenizer.json`, 32 MB) is over the 20 MB limit for git. Tests download it once from
`google/embeddinggemma-2` at `HF_REVISION` into `artifacts/tokenizer/<revision>/` and verify its sha256; the app gets
it from the published model folder through `ModelStore`.

## Contract model

`RuntimeTests` exercise the `EmbeddingGemma` actor on real Core ML before s4 exists, with a 52 KB stand-in that
has every entry point and I/O of spec section 7 and computes a vector the test recomputes exactly:

```
uv run --exact --group convert python -m stages.s5_swift.make_contract_model   # artifacts/s5_swift/contract/
```

Three extra functions (`test_nan_text_16`, `test_fp16_text_16`, `test_4d_text_16`) cover NaN outputs
(`.nonFiniteOutput`), FP16 outputs and a `[1, 768, 1, 1]` output. Without the package the suite is skipped with that
command as the message; with `EGCM_REQUIRE_ARTIFACTS=1` (CI) it fails instead.

## Known failure modes

- **Pixel test fails after an s0 rerun:** the committed samples are stale. Rerun `export_fixtures`.
- **JPEG pixel numbers look bad but PNG and synthetic are exact:** that is the JPEG decoder (ImageIO vs PIL), not
  the resize. See ADR-0004. It shows up again, if anywhere, in the end-to-end cosine on JPEG fixtures.
- **Tokenizer download fails:** the tests need network once (huggingface.co). After that the verified cache in
  `artifacts/tokenizer/` is used.
- **`max_pixel_abs_diff` is inf:** read `details.pixels.bucket_mismatches` and `details.missing`. A bucket mismatch
  means `Buckets.swift` and `egcm/buckets.py` disagree on that size; that is a bug, not a precision question.
- **`details.cosine` (vs s0) is low only on JPEG images while PNG is fine:** expected, and not gated (ADR-0008). ImageIO
  and PIL decode JPEGs slightly differently, and EmbeddingGemma 2 itself, FP32 included, is very sensitive on some
  images (the heavily blurred fixtures move to 0.86 with ±1/255 noise). `details.decoder` shows the same drop with no
  Swift and no compression involved. Before ADR-0008 this failed the stage at 0.9712 (i165, budget 280).
- **`min_cosine_runtime` fails:** the Swift call path disagrees with Python Core ML on identical inputs. That is a real
  bug in the Swift runtime (function choice, input type or layout, padding, compute units, truncation or
  normalization), not precision. `details.runtime.by_group` shows whether it is text, one budget or one format.
- **A task-level gate fails while `min_cosine_runtime` passes:** the model, not the port. Compare with s4's
  numbers. s5's are against the FP reference on Swift's pixels, so they differ from s4's only by the decoder's effect
  on quantization error.
- **`PixelDumpError`:** a PNG in `artifacts/s5_swift/pixels/` is missing or not byte-identical to the image Swift
  hashed. egcm-parity must run with `--pixels`. Stale files from an older run are caught by the sha256 check.
- **`fp_reference <budget>: computed N images, M differ from s0`:** the FP step and the parent disagree on which images
  need the FP32 rewrite. That is a bug in `fp_reference.needed` or `run.fp_needed`.
