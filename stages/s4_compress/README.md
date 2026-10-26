# s4_compress

Compressed variants of the FP16 model (`egcm/compress.py`); keeps the smallest that passes every s4 gate as
`artifacts/s4_compress/EmbeddingGemma2.mlpackage` (all 14 functions), then splits it into one package per image
budget (ADR-0010): `EmbeddingGemma2-b70.mlpackage` and `EmbeddingGemma2-b280.mlpackage`, each with the four text
functions and that budget's five image functions, and their device downloads `EmbeddingGemma2-b70.aar` and
`EmbeddingGemma2-b280.aar` (below), plus `checksums.json` (both archives, every file of the three packages).

Run: `just stage 4` (the `convert` group). Needs fresh passing s0 and s3.

## What it does

1. `egcm.parts.build(..., recipes=RECIPES)`: each function is converted to FP16 once, in its own process, and
   every recipe compresses that model into its own part; each part is merged into its recipe's package in its own
   process. coremltools 9 cannot compress multifunction models, so compression is per part and the merge
   deduplicates. That needs the same weight to compress to the same bytes in every part, so recipes are linear
   quantization only (k-means palettization needs scikit-learn or kmeans1d, which are not in the lock).
2. For each variant, smallest first: contract check, device archive (`egcm.archive.device_archive`: one Apple Archive,
   LZFSE, of the package and the tokenizer), package zip (size recorded only), compile, every fixture with
   `CPU_AND_NE`, Neural Engine placement, latency. The first variant that passes every gate wins.
3. Variants (`egcm.compress.RECIPES`): `int8_table4` only (int8 per-channel conv weights, int4 blocks of 32 for the
   table), the shipping recipe (ADR-0007). Only conv weights and the table are quantized; position embeddings, RoPE
   tables, the image affixes, norms and biases stay FP16. Measure-only (`MEASURE_ONLY`, never stage candidates):
   `fp16` (baseline), `vis4kv8_text8` (296 MB but min cosine 0.917 and int4 vision convs on the CPU), `int8`
   (about 410 MB).
4. **Split per budget** (`split.py`, ADR-0010), winner only. For each budget, in its own process under
   `egcm.parts`' memory guard: `extract` copies the budget's nine functions out of the 14-function package with
   `egcm.mfmerge.merge` (what coremltools' `MultiFunctionDescriptor` + `save_multifunction` do, as in
   `scripts/probes/extract_functions.py`, but streaming one weight at a time: the MIL of each function is copied
   unchanged, each distinct weight stored once). Then the function names and the I/O contract are checked and the
   device archive is built (inside it the package is named `EmbeddingGemma2.mlpackage`, whichever budget). Then
   `identity`, one more process: every function of both budget packages and the same function of the 14-function
   package on one deterministic input (`split.feed`), `CPU_ONLY`, one model loaded at a time; the output bytes must
   be equal, else the stage raises (status `error`) and writes no `checksums.json`. Result:
   `details.split_identity`. The 14-function package is kept (the Swift end-to-end tests and `s6 --package` use it,
   about 450 MB); its archive is deleted (it was only the variant's gated `download_mb`).
5. `--measure a,b,...` evaluates named variants without the stage runner (`measure/measurement.json`);
   `python -m scripts.measure_s4_eval a,b --placement a` does the same without the placement analysis except for
   the variants after `--placement` (`measure/fast.json`).

## Metrics

`egcm.ranking.metrics` (shared with s3; ADR-0006, ADR-0007), the worst of budgets 70 and 280:
- `min_cosine`: minimum over the 200 text vectors and the 400 bucketed image vectors, each against s0's vector for
  the same input (`emb`, `emb_bucket`).
- `min_recall_at_10`, `min_p5_spearman_top50`: the 120 app queries (`kind=query`) ranking the 200 images. The
  reference is s0's query vectors against s0's **bucketed** image vectors (`emb_bucket`, which s2 matches to 2e-8);
  the new side is our package on the same bucketed images. So they measure what FP16 and compression cost.
- Bucketing fidelity, against s0's **exact-aspect** vectors (`emb_raw`): `min_t2i_top1_in_top10` (share of queries
  whose exact-aspect top-1 image is in the package's top 10) and `max_i2i_label_p10_drop` (image→image category
  precision@10 of the exact-aspect reference minus the package's).
- `max_download_mb`: the device download, `egcm.archive.device_archive`: one Apple Archive (LZFSE) holding
  `EmbeddingGemma2.mlpackage/**` and `tokenizer/{tokenizer.json,tokenizer_config.json}`. For the winner it is the
  **larger of the two budget archives** (what any one phone downloads; ADR-0010). Choosing the winner uses each
  variant's 14-function archive (`variants.<recipe>.download_mb`), which is at least as large. The deflate zip of the
  package alone is recorded as `zip_mb`, not gated.
- `details.budgets.<budget>`: `package`, `device_archive`, `functions`, `package_mb`, `download_mb` (what
  `egcm.published.shipped` reads); `details.split_identity`: per budget and function, `identical` and
  `max_abs_diff`.
- Recorded only (`vs_raw`, `[s4_compress.targets]`): text→image recall@10 and p5 Spearman against exact-aspect;
  fixed buckets cannot reach 0.95 there (s0's own bucketed vectors give 0.855 / 0.877).

## Gate

`gates.toml [s4_compress]`: `min_cosine >= 0.99`, `min_recall_at_10 >= 0.95`, `min_p5_spearman_top50 >= 0.9`,
`min_t2i_top1_in_top10 >= 0.95`, `max_i2i_label_p10_drop <= 0.03`, `max_download_mb <= 360` (ADR-0007, accepted
2026-10-25).

## Known failure modes

- **Disk fills with Core ML E5 bundles.** Every function load writes a bundle of about one copy of its weights to
  `~/Library/Caches/python3/com.apple.e5rt.e5bundlecache`. While evaluating, `egcm.e5prune.BundlePruner` deletes the
  older ones; when the stage (or `scripts.measure_s4_eval`) exits, also after an error or SIGTERM, `egcm.e5exit`
  deletes every bundle the run created. Only `kill -9` leaves them behind.

- **Bucketing alone fails ranking against exact-aspect (resolved 2026-10-25, ADR-0007).** s0's own vectors,
  bucketed vs exact-aspect, give recall@10 0.855 (budget 70) / 0.877 (280) and p5 Spearman 0.22 / 0.67, so no recipe
  could pass the original gates. The ranking gates are now against the bucketed FP reference, and bucketing is gated
  at task level (`min_t2i_top1_in_top10`, `max_i2i_label_p10_drop`).
- **The merged package is about the sum of its parts**: deduplication failed (a recipe that is not deterministic).
- **"the budget packages differ from the 14-function package (CPU_ONLY)"**: the split changed a function. The
  extract copies each function's MIL and weights unchanged, so this is a bug in `split.py` or `egcm.mfmerge` (a
  weight pointed at the wrong offset), not precision. `details.split_identity` lists the functions and the largest
  difference. Nothing downstream can use the run (no `checksums.json`).
- **`stages.s4_compress.split extract failed (memory guard)`**: the extract passed `egcm.parts.FOOTPRINT_LIMIT_GB`.
  It holds one weight at a time, so look for another process using the memory.
- **Download over 360 MB**: `int8_table4`'s package zip was 350 MB (2026-10-24); the gated device archive adds the
  tokenizer (about 32 MB before compression) and uses LZFSE, so check `download_mb`, not `zip_mb`. `int8`, with the
  table at int8 too, is about 60 MB more. The table alone is 134 MB at int8 and 75 MB at int4 (blocks of 32).
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
  0.978 (32) / 0.988 (8). The table tolerates int4 (0.9975). Measured on all fixtures: `vis4kv8_text8` 296 MB, min
  cosine 0.917; `int8_table4` 350 MB, min cosine 0.991 (ADR-0007, "Measured variants").
- **Per-block int4 vision convs run on the CPU.** In `vis4kv8_text8` every image function places its 81 int4 vision
  convs (MLP, attention q/o, projector) and 16 GELUs on the CPU: `embed_image_70_square` takes 233 ms instead of 46.
  The op-count Neural Engine share still reads 0.992, so check the `off_ane` op types and the latency, not only
  `ane_fraction_min`.
