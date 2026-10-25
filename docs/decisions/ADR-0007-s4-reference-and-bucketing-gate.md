# ADR-0007: Gate s4 compression against the bucketed FP model, and bucketing separately at the task level

**Status:** accepted, 2026-10-25, founder approved, with the download limit raised to 360 MB and `int8_table4` as
the shipping recipe. See "Decision (accepted)" at the end; proposed 2026-10-24.

## Context

Spec §5, ADR-0002 and the plan define the s4 ranking gates (`min_recall_at_10 >= 0.95`, `min_p5_spearman_top50 >= 0.9`)
against s0's vectors for the **exact-aspect** images (`emb_raw`), so they include the cost of the ten fixed image
buckets. They are not reachable by any compression recipe: the uncompressed model already misses them.

### What bucketing costs (s0's own vectors: bucketed vs exact-aspect, no Core ML, no compression)

The bucketed FP reference is s0's `emb_bucket` (the official model on the bucket-resized image); s2's FP32 Core ML
package matches it to within 2e-8 cosine, so it stands for s2. 120 app queries (`kind=query`) ranking 200 images.

| metric | budget 70 | budget 280 |
|---|---|---|
| min cosine, bucketed vs exact-aspect image vector | 0.916 | 0.893 |
| text→image recall@10 | 0.855 | 0.877 |
| text→image recall@5 / recall@1 | 0.815 / 0.767 | 0.858 / 0.900 |
| text→image NDCG@10, binary (mean / 5th percentile) | 0.892 / 0.715 | 0.912 / 0.791 |
| text→image NDCG@10, graded by exact-aspect rank (mean / p5) | 0.916 / 0.732 | 0.946 / 0.830 |
| text→image p5 Spearman ρ of the top 50 | 0.217 | 0.668 |
| exact-aspect top-1 found in the bucketed top 10 | 98.3% | 100% |
| median score gap between exact-aspect ranks 10 and 11 | 0.0013 | 0.0013 |
| median score spread across the exact-aspect top 10 | 0.039 | 0.040 |
| image→image precision@10 by category label, exact-aspect | 0.521 | 0.487 |
| image→image precision@10 by category label, bucketed | 0.510 | 0.492 |

The texts have no ground-truth labels, so text→image can only be compared with the exact-aspect ranking. The images
have category labels (14 categories, 10 to 30 images each); by those, bucketing does not change retrieval quality
(-0.011 at 70, +0.005 at 280). The overlap metrics are dominated by near ties: the 10th and 11th results differ by
0.0013 in score, against a 0.04 spread across the top 10, so any perturbation reorders them. Even an image whose aspect
matches its bucket within 2% has a cosine as low as 0.916 to its exact-aspect vector, because the processor's
exact-aspect resize gives a different resolution and token count than the bucket. A few more buckets would therefore
not lift text→image recall@10 to 0.95; only exact-grid functions (one per processor output size) would.

## Decision (as proposed 2026-10-24)

Split the s4 gates into two families with separate references.

**(a) Compression gates, against the bucketed FP model** (s0 `emb_bucket` = s2 to 2e-8). Same keys and thresholds:
`min_cosine >= 0.99`, `min_recall_at_10 >= 0.95`, `min_p5_spearman_top50 >= 0.9`, `max_download_mb <= 300`. They
isolate what compression (and FP16) costs.

**(b) Bucketing fidelity, against s0's exact-aspect vectors, at the task level**, as separate keys. Text→image
recall@10 measures 0.855 / 0.877, below 0.9, but the label-based measure shows no quality loss and the exact-aspect top
result stays in the bucketed top 10 for 98 to 100% of queries. Proposed:
- `min_t2i_top1_in_top10 >= 0.95`: the exact-aspect top-1 image is in the package's top 10, both budgets.
- `max_i2i_label_p10_drop <= 0.03`: image→image category precision@10 of the package is at most 0.03 below that of
  the exact-aspect reference, both budgets.
- Text→image recall@10 and NDCG@10 against exact-aspect recorded as targets (`[s4_compress.targets]`), not gated.

If the coordinator prefers to keep text→image recall@10 against exact-aspect as a gate, it cannot pass with fixed
buckets; the alternative is exact-grid functions (one per processor output shape), which this ADR does not propose
because of the package size and the number of functions.

## Measured variants (s4 measure, FP16 + compression, `CPU_AND_NE`)

`python -m scripts.measure_s4_eval vis4kv8_text8,fp16,int8_table4 --placement vis4kv8_text8` at commit 77d1241, plus the
placement of `int8_table4` (M1 Pro, macOS 26), all 600 fixture vectors on `CPU_AND_NE`;
`artifacts/s4_compress/measure/fast.json`. Ranking: the 120 app queries over the 200 images, budget 70 / 280. Latency:
median of 20 after warm-up. ANE: share of ops placed on the Neural Engine by `MLComputePlan`, minimum over the 14
functions.

| | `fp16` (baseline) | `vis4kv8_text8` | `int8_table4` |
|---|---|---|---|
| download (zip) / package, MB | 700.2 / 934.8 | **295.6** / 394.2 | 349.7 / 452.4 |
| min cosine, text (200) | 0.99996 | 0.99791 | 0.99791 |
| min cosine, image (400), vs bucketed FP | 0.99404 | **0.91694** | 0.99133 |
| recall@10 vs bucketed FP | 0.968 / 0.983 | 0.861 / 0.881 | 0.962 / 0.966 |
| p5 Spearman top-50 vs bucketed FP | 0.987 / 0.995 | 0.562 / 0.693 | 0.966 / 0.972 |
| recall@10 vs exact-aspect (current gate) | 0.843 / 0.878 | 0.792 / 0.838 | 0.844 / 0.874 |
| p5 Spearman top-50 vs exact-aspect (current gate) | 0.231 / 0.655 | -0.108 / 0.349 | 0.169 / 0.681 |
| latency `embed_text_16` / `embed_image_70_square`, ms | 6.5 / 47.9 | 4.5 / **233.0** | 4.4 / 45.8 |
| ANE share (ops), min over functions | 0.99864 (s3) | 0.99240, but 81 convs + 16 GELUs per image function on the CPU | 0.99813 (text 5873/5884; images 16851/16853, as FP16) |

Per image function, `vis4kv8_text8` minimum cosine 0.917 (280_portrait) to 0.988 (280_tall); `int8_table4` 0.9913
(280_landscape) to 0.9995 (280_tall).

Reading it:
- Under the current gates (ranking against exact-aspect) nothing passes, FP16 included, as above.
- Under (a): `fp16` passes everything except the download; `int8_table4` passes min cosine, recall@10 and p5
  Spearman but downloads 350 MB, 50 MB over `max_download_mb`; `vis4kv8_text8`, the smallest variant that runs on the
  Neural Engine, fits (296 MB) and fails the rest. So with deterministic linear quantization no variant passes (a)
  at 300 MB; that is a size versus accuracy decision for the founder, not something a recipe can fix.
- `vis4kv8_text8` also loses the speed: its per-block int4 vision convs (MLP, attention q/o, projector) are placed on
  the CPU in every image function, so `embed_image_70_square` takes 233 ms instead of 46. The op-count share (0.992)
  hides this; s3's `min_ane_fraction` would pass it.

### Why no linear recipe reaches both 0.99 and 300 MB

Parameters above the 2,048 threshold: table 134.2M, text tower 136.7M, vision tower 152.5M (attention 37.8M, MLP
113.3M). Per parameter, int8 per-channel costs 1 byte, int4 per-block (32) 0.5625; the zip saves about 11% at int8.
Screens on single-function FP16 parts on `CPU_ONLY` (compression error only; 78 texts of `embed_text_32`, 12 images
of `embed_image_70_tall`; minimum cosine against s0):

| change | min cosine |
|---|---|
| text tower + table int4 b32 | 0.974 (text) |
| text tower int4 b16, table int4 b32 | 0.982 (text) |
| text tower int8, table int4 b32 / int4 per-channel | 0.9975 / 0.9943 (text) |
| text tower int8, table int8 | 0.9997 (text) |
| vision MLP int4 b32 / b16 / b8 / per-channel, rest int8 | 0.978 / 0.981 / 0.988 / 0.963 (image) |
| everything int8 | 0.995 to 0.998 (image; CPU runs vary by about 0.003) |
| one sub-block at int4 b32, rest int8 (text: table int4) | text: q 0.9973, k 0.9967, v 0.9942, o 0.9950, MLP gate / up / down 0.9950 / 0.9951 / 0.9912, PLE gate / proj 0.9957 / 0.9946; image: vision MLP gate / up / down 0.989 / 0.990 / 0.987, vision o 0.990, vision q 0.995, text MLP gate / up / down 0.993 / 0.992 / 0.991, text o 0.991 |

`int8_table4` is 50 MB over; closing that takes int4 instead of int8 on about 130M more parameters (0.44 byte saved
each), and every sub-block worth more than 6 MB costs 0.003 to 0.008 on the worst image alone, on top of the 0.9913
already measured. Uniform 6-bit palettization (deterministic, no new dependency, but not linear quantization) was
screened on the text tower too: per tensor 0.949, per 16 channels 0.989 (33 MB less than int8), per channel 0.996 (7 MB
less). The first int4 package (`int4b32`, all constants int4, 215 MB zipped) would also have failed: its text functions
screen at min cosine 0.974, and it could not load its image functions on the Neural Engine (next section).

## ANE compile (2026-10-24)

The first measured int4 package (`int4b32`: int4 blocks of 32 on every constant above 2,048 elements, the RoPE
tables and position embeddings included) ran every text function on `CPU_AND_NE`, then failed at the first image
function with `RuntimeError: MLModelConfiguration's .functionName property must be nil unless the model type is ML
Program`. The real error is on stdout just before it: `E5RT ... MILCompilerForANE error: failed to compile ANE model
using ANEF. Error=_ANECompiler : ANECCompile() FAILED`. The Neural Engine compiler rejected the function, and Core
ML's fallback cannot load one function of a multifunction package. A single-function model in the same state loads
anyway through the fallback (slowly: 300 to 485 s instead of 75 to 190 s), so a successful load proves nothing; the
verdicts below come from the E5RT message (C++ output, flushed when the process exits).

Bisection on single-function FP16 parts (`python -m egcm.parts convert --recipes fp16`), compressed with
coremltools 9 `linear_quantize_weights` (`op_name_configs`, so only the named weights change), compiled, loaded with
`CPU_AND_NE`, one fixture predicted. int4 means per-block, block 32, symmetric, unless noted; everything not listed
stays FP16.

| part | int4 on | int8 per-channel on | ANE compile |
|---|---|---|---|
| embed_image_70_tall | every conv weight (tables FP16) | | **failed** |
| embed_image_70_tall | vision MLP (48 weights) | | ok |
| embed_image_70_tall | vision attention q, k, v, o (64) | | **failed** |
| embed_image_70_tall | vision attention q / k / v / o alone (16 each) | | ok / **failed** / **failed** / ok |
| embed_image_70_tall | vision attention, block 64 | | **failed** |
| embed_image_70_tall | vision attention, per-channel int4 | | ok |
| embed_image_70_tall | | vision attention | ok |
| embed_image_70_tall | text tower inside the image function (218) | | ok |
| embed_image_70_tall | vision projector | | ok |
| embed_image_70_tall | every conv weight except vision k/v | vision k/v | ok |
| embed_image_280_tall | every conv weight except vision k/v | vision k/v | **failed** |
| embed_image_280_tall | text tower and vision MLP | vision attention | **failed** |
| embed_image_280_tall | text tower | vision tower | **failed** |
| embed_image_280_tall | text tower only | | **failed** |
| embed_image_280_tall | vision MLP only | | ok |
| embed_image_280_tall | vision attention q, o | vision k/v | ok |
| embed_image_280_tall | vision tower except k/v | vision k/v, text tower | ok |
| embed_image_280_tall | | every conv weight | ok |

So two things fail: per-block int4 on the vision attention k and v projections (at budget 70, and so at 280), and
per-block int4 on the text tower inside the budget-280 image functions (about 280 image tokens plus the affixes go
through it; the same weights compile in the text functions, up to 128 tokens, and in the budget-70 image functions).
int8 compiles everywhere. The compiler gives no detail beyond `ANECCompile() FAILED`, so the shape or size limit
behind it is not known. The patch-embedding conv (C_in = 3) was never the problem: per-block int4 cannot split it and
coremltools leaves it FP16 ("Cannot perform quantization on vision_encoder_patch_weight_to_fp16").

The recipes in `egcm/compress.py` (commits 8c0073d, 77d1241) therefore quantize only conv weights and the
embedding table, chosen by the consuming op (position embeddings, RoPE tables, affixes, norms and biases stay FP16;
29 MB over the 14 functions), and the int4 recipe `vis4kv8_text8` keeps the vision k/v projections and the text tower
at int8. `embed_image_280_tall` with exactly that recipe compiles and runs on the Neural Engine.

## Decision (accepted 2026-10-25, founder approved)

(a) and (b) as proposed above, and the download limit raised from 300 to 360 MB: no deterministic linear recipe
reaches the cosine gate at 300 MB ("Measured variants"), and `int8_table4` passes (a) at 350 MB with Neural Engine
speed. `gates.toml [s4_compress]` changes in the same commit:

| key | before | now |
|---|---|---|
| `min_cosine` | 0.99 | 0.99 |
| `min_recall_at_10` | 0.95 against exact-aspect | 0.95 **against the bucketed FP reference** |
| `min_p5_spearman_top50` | 0.9 against exact-aspect | 0.9 **against the bucketed FP reference** |
| `min_t2i_top1_in_top10` | | **0.95** (new, (b): the exact-aspect top-1 image in the package's top 10, worst budget) |
| `max_i2i_label_p10_drop` | | **0.03** (new, (b): image→image category precision@10, exact-aspect reference minus package, worst budget) |
| `max_download_mb` | 300, the deflate zip of the package | **360, the device download: one Apple Archive (LZFSE) of the package and the tokenizer** |
| `[s4_compress.targets]` | | `t2i_recall_at_10_exact_aspect = 0.95` (recorded, not gated) |

The metrics are `egcm.ranking` (s3 uses the same module, ADR-0006); `stages/s4_compress/run.py` checks every gate in
`gates.toml` per variant. Of the proposed targets only text→image recall@10 against exact-aspect is recorded: NDCG@10
was a one-off analysis and has no agreed target.

Recipes (`egcm/compress.py`): `int8_table4` is the only stage candidate. `vis4kv8_text8` and `int8` are measure-only
(`MEASURE_ONLY`, still buildable with `scripts.measure_s4_eval`): `vis4kv8_text8` fails min cosine (0.917) and runs its
int4 vision convs on the CPU (233 ms per image), and `int8` is about 410 MB, over the limit. Neither can win, and each
would cost the stage about two hours of evaluation and placement before `int8_table4`.

### What `max_download_mb` measures (2026-10-25)

The 360 MB approval is for what an iPhone downloads. The app publishes and downloads one Apple Archive with LZFSE
compression (`egcm.archive.device_archive`, built with `aa archive ... -a lzfse`, unpacked on iOS with the
AppleArchive framework), holding exactly `EmbeddingGemma2.mlpackage/**` and `tokenizer/tokenizer.json`,
`tokenizer/tokenizer_config.json` (the files the Swift tokenizer reads); no symlinks, deterministic order. The gate
measures that file, tokenizer included; the deflate zip of the package alone, which the measurements above used
(`int8_table4` 349.7 MB), is still recorded as `zip_mb` but not gated. s4 writes the archive as
`artifacts/s4_compress/EmbeddingGemma2.aar`, with its sha256 in `checksums.json`. The limit stays 360 MB; if
`int8_table4` measured over it, that goes back to the founder, not into this file.

