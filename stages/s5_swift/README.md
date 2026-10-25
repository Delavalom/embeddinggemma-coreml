# s5_swift

The Swift package on macOS against s0: token ids, preprocessed pixels, and end-to-end vectors.

Run: `just stage 5` (the `convert` group). Needs fresh passing s0 and s4. The package is the one s4's report says it shipped
(`egcm.published.shipped`: `details.package`, recipe `details.winner`, recorded here as `details.s4_recipe`).

## What it does

1. `run.py` writes the synthetic pattern (`export_fixtures.synthetic`) at `SYNTHETIC_SIZES` as PNGs into
   `artifacts/s5_swift/synthetic/`, with torch's uint8 antialiased bicubic of each as the reference.
2. It runs `swift run -c release egcm-parity` (`swift/Sources/egcm-parity/`). The CLI goes through the path an app
   takes: it puts the package and the tokenizer (s0's HF snapshot, else `artifacts/tokenizer/<revision>/`,
   sha256-checked by `egcm.published.tokenizer_source`) in a
   local published folder with `files.json`, installs it with `ModelStore` (verify, compile) and loads
   `EmbeddingGemma.load(.file(...), dimensions: 768)` once per budget, on `.cpuAndNeuralEngine`. For every fixture
   text (prompt by kind: `query` or `document`) it records the token ids and the vector; for every fixture image and
   each synthetic PNG at budgets 70 and 280 it records the bucket, the sha256 of the uint8 bucket image and 8,192
   pixel values (the `export_fixtures` sampling rule), and for fixture images the vector. Output:
   `artifacts/s5_swift/parity.json`.
3. `compare.py` (pure functions, `tests/test_s5_compare.py`) compares that with s0 and the torch references.

## Metrics

- `token_ids_identical`: every one of the 200 texts has exactly s0's ids (`ids_<id>` in `texts.npz`), none missing.
- `max_pixel_abs_diff`: the 99th percentile of `|swift - reference|` (0-1 units) over the pixel samples of the
  decode-identical inputs (ADR-0004): the PNG fixtures against s0's stored bucket pixels and the synthetic PNGs
  against torch. It is `inf` if any image lands in another bucket than the reference or an output is missing.
  `details.pixels` also has the JPEG numbers (`jpeg_pixel_p99`, decoder included, recorded only), the maxima, and the
  bit-exact counts per group.
- `min_cosine`: the minimum over the 200 text vectors and the 400 image vectors (both budgets), each against s0's
  vector for the same input (`emb`, `emb_bucket`). `details.cosine` splits it by text kind and by budget and format,
  with the five worst ids of each.

## Gate

`gates.toml [s5_swift]`: `token_ids_identical == true`, `max_pixel_abs_diff <= 0.003922`, `min_cosine >= 0.99`.

## Measuring a package without a report

```
uv run --exact --group convert python -m stages.s5_swift.run --measure <pkg> [--s0 <dir>] [--compute-units cpuOnly] [--limit N]
```

runs the same comparison on any package with the spec section 7 I/O and prints the metrics and details as JSON
(files go to `artifacts/s5_swift/measure/`). On the contract model (below), 2026-10-24, M1 Pro: ids identical for
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
- **Cosine fails only on JPEG images (`details.cosine.image.*_jpeg`) while PNG passes:** the decoder (ADR-0004).
  Compare with s4's cosine for the same ids, which used PIL's decode: if s4 passes them, it is ImageIO vs PIL.
- **Cosine fails on texts as well:** compare `details.compute_units` and s4's `CPU_AND_NE` numbers; s5 runs the same
  package on the same units, so a gap there points at the Swift runtime (inputs, function choice), not the model.
