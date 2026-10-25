# s4_compress

Compressed variants of the FP16 model (`egcm/compress.py`); keeps the smallest that passes every s4 gate as
`artifacts/s4_compress/EmbeddingGemma2.mlpackage` (+ `.zip` and `checksums.json`).

Run: `just stage 4` (the `convert` group). Needs fresh passing s0 and s3.

## What it does

1. `egcm.parts.build(..., recipes=RECIPES)`: each function is converted to FP16 once, in its own process, and
   every recipe compresses that model into its own part; each part is merged into its recipe's package in its own
   process. coremltools 9 cannot compress multifunction models, so compression is per part and the merge
   deduplicates. That needs the same weight to compress to the same bytes in every part, so recipes are linear
   quantization only (k-means palettization needs scikit-learn or kmeans1d, which are not in the lock).
2. For each variant, smallest first: contract check, zip (the download), compile, every fixture with `CPU_AND_NE`,
   Neural Engine placement, latency. The first variant that passes every gate wins.
3. Variants (`egcm.compress.RECIPES`, smallest first): `vis4kv8_text8` (int4 blocks of 32 for the vision tower and
   the table; vision attention k/v and the text tower at int8), `int8_table4` (int8 per-channel conv weights, int4 table), `int8` (int8
   conv weights and table). Only conv weights and the table are quantized; position embeddings, RoPE tables, the
   image affixes, norms and biases stay FP16. `fp16` is a measure-only baseline (`--measure`).
4. `--measure a,b,...` evaluates named variants without the stage runner (`measure/measurement.json`);
   `python -m scripts.measure_s4_eval a,b --placement a` does the same without the placement analysis except for
   the variants after `--placement` (`measure/fast.json`).

## Metrics

- `min_cosine`: minimum over the 200 text vectors and the 400 bucketed image vectors, each against s0's vector for
  the same input (`emb`, `emb_bucket`).
- `min_recall_at_10`, `min_p5_spearman_top50`: the 120 app queries (`kind=query`) ranking the 200 images, at budget
  70 and at 280 (the minimum of the two). The reference is s0's query vectors against s0's **exact-aspect** image
  vectors (`emb_raw`, spec §5, ADR-0002); the new side is our package on the bucketed images. So the gate includes the
  cost of bucketing. The same metrics against s0's bucketed vectors are recorded as `vs_bucket` (compression cost
  alone); they are not gated.
- `max_download_mb`: the zipped package.

## Gate

`gates.toml [s4_compress]`: `min_cosine >= 0.99`, `min_recall_at_10 >= 0.95`, `min_p5_spearman_top50 >= 0.9`,
`max_download_mb <= 300`.

## Known failure modes

- **Bucketing alone fails the ranking gates (open, 2026-10-23).** s0's own vectors, bucketed vs exact-aspect, give
  recall@10 0.855 (budget 70) / 0.877 (280) and p5 Spearman 0.22 / 0.67, and min cosine 0.916 / 0.893. Even images
  whose aspect matches their bucket within 2% have cosines down to 0.916, because the processor's exact-aspect resize
  uses a different resolution and token count than the bucket. So no compression recipe can pass the gates as defined,
  and adding a few buckets would not close the gap. Resolving it needs a decision recorded in an ADR (the reference
  set or the bucket design); gates are not changed without one.
- **The merged package is about the sum of its parts**: deduplication failed (a recipe that is not deterministic).
- **Download over 300 MB**: int8 everywhere is about 440 MB; the table alone is 134 MB at int8 and 67 MB at int4.
- **`MLModelConfiguration's .functionName property must be nil unless the model type is ML Program`** when loading an
  image function on `CPU_AND_NE`, preceded on stdout by `E5RT ... MILCompilerForANE error ... ANECCompile() FAILED`:
  the Neural Engine compiler rejected the function and Core ML's fallback cannot load one function of a
  multifunction package. A single-function model in the same state loads silently through the fallback (slowly), so
  check the log for `ANECCompile() FAILED` (C++ output, flushed when the process exits). Causes found 2026-10-24:
  per-block int4 (block 32 or 64) on the vision attention k or v weights (any budget; per-channel int4 and int8
  compile, q and o compile at per-block int4), and per-block int4 on the text tower inside the budget-280 image
  functions (it compiles in the text functions and at budget 70). Bisection: ADR-0007, "ANE compile".
- **int4 costs more than the cosine gate allows** (2026-10-24, single-function screens against s0): int4 on the
  text tower gives min text cosine 0.974 (blocks of 32) / 0.981 (16); int4 on the vision MLP gives min image cosine
  0.978 (32) / 0.988 (8). The table tolerates int4 (0.9975). int8 conv weights with an int4 table hold about 0.995
  but zip to about 330 MB (ADR-0007, "Measured variants").
