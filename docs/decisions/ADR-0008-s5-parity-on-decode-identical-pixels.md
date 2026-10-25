# ADR-0008: s5 measures the Swift runtime on decode-identical pixels, and the task level as s4 does

**Status:** Proposed, 2026-10-25. **Not accepted.** `gates.toml` is unchanged. The founder decides; until then
`[s5_swift] min_cosine >= 0.99` stands and s5 stays failed.

## Context

s5 (`just stage 5`, 2026-10-25) failed only `min_cosine`: 0.9712 against `>= 0.99`. It runs the shipped s4 package
(`int8_table4`, from the device archive) through the Swift package on `cpuAndNeuralEngine` and compares every vector
with s0. Token ids are identical (200/200). PNG and synthetic pixels are bit-exact (60/60, 20/20). Text min cosine is
0.9979 and PNG images are at least 0.9954. Only JPEG images fall below 0.99: 280 i165 0.9712, i154 0.9803, i161
0.9834; 70 i092 0.9789.

The JPEGs are the one input where s5 and s4 differ. s4 feeds s0's stored bucket pixels (PIL decode, then torch's
resize) to Python Core ML. Swift decodes with ImageIO and then runs the same resize (`Resampler.swift`, bit-exact with
torch on lossless input, ADR-0004). ADR-0004 left the JPEG decoder's effect to this cosine gate and said: "the fix
would be a different decoder, not a looser cosine gate". ADR-0006 and ADR-0007 later replaced per-image cosine with
task-level gates at s3 and s4, because a few fragile images cannot reach 0.995. s5's gate was not revisited.

## What was measured (M1 Pro, macOS 26, 2026-10-25)

Ids: the 10 worst s5 JPEGs (i165, i092, i154, i161, i150, i131, i151, i141, i001, i179) and 5 at the s5 median
(i156, i008, i036, i166, i024), at both budgets, so 30 cases. One variable changes per comparison.

**How the pixels were made.**
- *PIL:* `egcm.fixtures.open_image` (s0's decode), saved as lossless PNG. torch's resize of it equals s0's stored
  `pixels_<id>` in 30/30 cases. egcm-parity's sha256 of its bucket image equals s0's in 30/30 cases, so Swift fed the
  model exactly s0's (and s4's) bucket pixels.
- *ImageIO:* a standalone Swift tool runs `CGImageSourceCreateWithURL`, then `CGImageSourceCreateImageAtIndex(0,
  nil)`, then draws into an 8-bit sRGB `noneSkipLast` `CGContext` (`interpolationQuality = .none`, blend `.copy`,
  the same code as `RowProvider.draw` for opaque images) and saves as PNG. torch's resize of that equals the bucket
  image Swift used for the original JPEG in s5 (parity.json sha256) in **340/340** JPEG cases. So these are exactly
  the pixels the Swift package fed the model. All fixture JPEGs are untagged; ImageIO reports `kCGColorSpaceSRGB`,
  so no colour conversion happens.

Swift runs used `swift run -c release egcm-parity` on a fixtures folder holding each id three times: the PIL PNG,
the ImageIO PNG and the original JPEG. Python Core ML used the same unpacked package, compiled, `CPU_AND_NE`,
`egcm.e5cache.PurgingPackage` (as s4). "FP32" is the s1 rewrite in PyTorch FP32 on CPU, built as in
`stages/s1_rewrite` (it matches s0 to cosine 1 - 6e-9 on these images).

### 1. The Swift runtime against Python Core ML on identical pixels (hypothesis: the Swift port has a bug)

| comparison (30 cases) | min cosine |
|---|---|
| (a) Swift (PIL PNG) vs Python Core ML (PIL pixels) | **1.000000000** (max element diff 4.5e-8) |
| (a') Swift (original JPEG) vs Python Core ML (ImageIO pixels) | **1.000000000** |
| Swift (ImageIO PNG) vs Swift (original JPEG) | 1.000000 |
| Swift (JPEG) now vs the s5 run | 1.000000 |
| Python Core ML, the same input twice | 1.000000 |

Rejected: on the same pixels, the Swift call path (function choice, input type, compute units, truncation and
normalization) gives the same vector as Python Core ML. Text vectors already matched s4 to about 1e-10 (t011
0.99887418610 vs 0.99887418618). The runtime is deterministic.

### 2. Per id: what changes the cosine against s0 (worst cases; all 30 are in the investigation log)

| id | budget | (b) Swift (PIL) vs s0 | s4 (Python, PIL) vs s0 | (c) Swift (JPEG) vs s0 = s5 | decoder only, FP32: ImageIO vs PIL | decoder only, int8: ImageIO vs PIL | Swift (JPEG) vs FP32 (ImageIO) |
|---|---|---|---|---|---|---|---|
| i165 | 280 | 0.9934 | 0.9934 | **0.9712** | **0.9716** | 0.9747 | 0.9970 |
| i154 | 280 | 0.9964 | 0.9964 | **0.9803** | **0.9806** | 0.9822 | 0.9942 |
| i161 | 280 | 0.9913 | 0.9913 | **0.9834** | 0.9915 | 0.9917 | 0.9902 |
| i150 | 280 | 0.9943 | 0.9943 | 0.9904 | 0.9948 | 0.9962 | 0.9947 |
| i131 | 280 | 0.9998 | 0.9998 | 0.9924 | 0.9920 | 0.9926 | 0.9997 |
| i151 | 280 | 0.9992 | 0.9992 | 0.9933 | 0.9946 | 0.9945 | 0.9992 |
| i001 | 280 | 0.9997 | 0.9997 | 0.9941 | 0.9947 | 0.9950 | 0.9997 |
| i092 | 70 | 0.9997 | 0.9997 | **0.9789** | **0.9814** | 0.9800 | 0.9996 |
| i141 | 70 | 0.9932 | 0.9932 | 0.9935 | 0.9942 | 0.9997 | 0.9986 |
| i179 | 70 | 0.9991 | 0.9991 | 0.9944 | 0.9956 | 0.9953 | 0.9990 |
| 5 median ids | both | ≥ 0.9995 | ≥ 0.9995 | ≥ 0.9992 | ≥ 0.9995 | ≥ 0.9995 | ≥ 0.9993 |
| **min of 30** | | 0.9913 | 0.9913 | 0.9712 | 0.9716 | 0.9747 | 0.9902 |

- With PIL pixels, Swift reproduces s4 per image to 4 decimals. Its floor is s4's (0.9913, i161 280).
- Changing only the decoder moves the **FP32 reference model itself** by as much as s5 measured: i165 0.9716 (s5
  0.9712), i154 0.9806 (0.9803), i092 0.9814 (0.9789). Neither quantization, the Neural Engine nor Swift is
  involved in that column.
- On the same (ImageIO) pixels, Swift is ≥ 0.9902 against FP32: what quantization and FP16 on the Neural Engine
  cost, the same as s4 on PIL pixels.

### 3. Sensitivity: every value moved by exactly 1/255 (random sign, 3 seeds), Python Core ML and FP32

| id, budget | int8 Core ML (`CPU_AND_NE`) | FP32 (s1 rewrite) | ImageIO vs PIL, int8 / FP32 |
|---|---|---|---|
| i161 280 (blurry) | 0.866 / 0.864 / 0.865 | 0.866 / 0.865 / 0.865 | 0.9917 / 0.9915 |
| i165 280 (blurry) | 0.907 / 0.896 / 0.897 | 0.903 / 0.892 / 0.893 | 0.9747 / 0.9716 |
| i154 280 (blurry) | 0.934 / 0.937 / 0.933 | 0.938 / 0.941 / 0.938 | 0.9822 / 0.9806 |
| i008 280 (people) | 0.988 / 0.990 / 0.981 | 0.988 / 0.990 / 0.982 | 0.9996 / 0.9996 |
| i024 280 (dogs) | 0.995 / 0.994 / 0.996 | 0.995 / 0.994 / 0.996 | 0.9998 / 0.9998 |
| i092 70 (cars) | 0.996 / 0.989 / 0.999 | 0.996 / 0.989 / 0.999 | 0.9800 / 0.9814 |

The fragility belongs to EmbeddingGemma 2's vision tower itself. Quantization adds nothing measurable: the FP32
and int8 columns agree to about 0.004. One level of noise per value moves the FP32 vector of the heavily blurred
fixtures (`blur_radius` 7 to 10) to 0.86–0.94. This also explains why per-image pixel difference does not predict the
cosine drop (coordinator: corr 0.014 over 400 cases). The decoder differences on the blurry images are tiny (max 3 to
6/255), but on those images the model amplifies them. On textured photos it does not, even at 23 to 55/255. Why the
tower behaves this way on low-texture inputs was not investigated (not understood).

### 4. Task level on the Swift vectors (`python -m scripts.s5_task_metrics`, plus the same `egcm.ranking.metrics`
with the reference swapped)

| metric | s4 gate | s4 (Python, PIL) vs s0 | Swift vs s0 (s5 as is) | decoder only: FP32 (ImageIO) vs s0 | Swift vs FP32 (ImageIO) |
|---|---|---|---|---|---|
| min cosine (text + image) | ≥ 0.99 | 0.9913 | **0.9712** | 0.9716 | 0.9902 |
| p1 cosine, image | (s3: ≥ 0.995) | 0.9954 | 0.9904 | 0.9920 | 0.9968 |
| recall@10 vs bucketed reference, worst budget | ≥ 0.95 | 0.9617 | **0.9467** (70: 0.9500) | 0.9683 | 0.9575 |
| p5 Spearman top-50, worst budget | ≥ 0.9 | 0.9660 | 0.9528 | 0.9691 | 0.9645 |
| t2i top-1 in top-10 (vs exact-aspect) | ≥ 0.95 | 0.9833 | 0.9833 | 0.9833 | 0.9833 |
| i2i label p@10 drop (vs exact-aspect) | ≤ 0.03 | 0.0120 | 0.0065 | 0.0045 | 0.0065 |

"FP32 (ImageIO)" uses the FP32 vectors for the 170 JPEGs and s0's `emb_bucket` for the 30 PNGs (identical decode).
Measured against s0's PIL reference, even s4's own ranking gate would fail on the Swift vectors at budget 280
(0.9467). The decoder alone costs about 3 points of recall@10 in FP32, and compression about 4 more. Against a
reference computed on the pixels Swift actually used, every s4 gate passes, with the same margins as s4.

## Root cause

The decoder changes the input, and a few inputs make the model fragile. ImageIO and PIL decode the fixture JPEGs
slightly differently (the coordinator's black-box decode diffs: ImageIO matches libjpeg-turbo without fancy chroma
upsampling; PIL uses fancy upsampling). On heavily blurred images, EmbeddingGemma 2's vision tower is fragile, FP32
included: ±1/255 moves it to 0.86. So s0's reference vector for those images depends on the decoder, and an exact
port cannot reach 0.99 against a PIL-decoded reference with ImageIO's pixels. The Swift port is correct (cosine 1 to
Python Core ML on identical pixels). Quantization does not amplify the decoder difference: over the 30 cases, the int8 package
never moves more than 0.0015 beyond FP32 (i092 70), and once moves 0.0054 less (i141 70).

Confidence: high. Every link was measured with one variable changed, and the ImageIO pixels are verified to be
exactly what the Swift package used (sha256, 340/340). Not established: the mechanism inside the vision tower.

This contradicts ADR-0004's expectation that the fix would be a different decoder. Matching PIL's decoder would
make this fixture comparison pass, but it would not make the embedding more correct. An app photo is a HEIC or JPEG
decoded by ImageIO, and no PIL reference exists for it.

## Options

1. **Keep the gate. Fix the port.** No port bug was found, so there is nothing to fix. s5 stays failed under the
   current gate.
2. **Match the decoder.** Ship a libjpeg-turbo-compatible JPEG decoder (fancy upsampling) in the Swift package, or
   decode the s5 fixtures with it. This needs a vendored C library and a licence review, adds binary size, and
   covers only JPEG (HEIC still goes through ImageIO). It optimizes the test, not the product. Not recommended.
3. **Make the decoder disappear from the fixture.** Convert the s0 JPEG fixtures to PNG, or have egcm-parity read
   PIL-decoded PNGs. It passes today (the Swift (PIL) column: min 0.9913, which is s4's). But the gate then never
   sees the decoder the app uses, and changing fixtures means rerunning s0 to s4.
4. **(Recommended) Split s5 like ADR-0006/0007.** Use one reference per question, and add no new tolerance:
   - **Runtime parity, decode-identical:** `min_cosine` becomes Swift vs Python Core ML (the same s4 package, the
     same compute units) on the bucket pixels the Swift package produced, for all 600 vectors. Measured 1.000000000
     on 30 cases. A threshold of 0.9999 is far stricter than today's 0.99 and catches any runtime or preprocessing
     regression.
   - **The s4 gates on the Swift vectors**, against the FP reference computed on Swift's pixels (FP32 rewrite or s2
     FP32 Core ML, which matches s0 to 2e-8; PNGs can reuse s0): `min_cosine_image >= 0.99`,
     `min_recall_at_10 >= 0.95`, `min_p5_spearman_top50 >= 0.9`, `min_t2i_top1_in_top10 >= 0.95`,
     `max_i2i_label_p10_drop <= 0.03`. These are s4's thresholds unchanged. Measured: 0.9902, 0.9575, 0.9645, 0.9833,
     0.0065. The image floor margin is thin: 0.0002 at i161 280 (s4 had 0.0013 on the same image
     with PIL pixels).
   - **Recorded, not gated:** the decoder's own cost, FP32 on ImageIO vs PIL pixels (min cosine 0.9716, recall@10
     0.9683), and today's numbers against s0 (0.9712, 0.9467). That keeps the decoder effect visible.
   - Cost: egcm-parity must write its full uint8 bucket images (today it writes a sha256 and 8,192 samples), so s5
     can feed them to Python Core ML and the FP reference. s5 then runs one Python Core ML pass (tens of minutes, mostly ANE
     compilation; the 30-case Swift run here took 27 min) and one FP pass on the 340 JPEG cases (about 21 min on CPU, 6 threads). It runs
     one heavy process at a time, as now.

## Recommendation

Option 4. It changes what s5 measures in the same way ADR-0006/0007 changed s3/s4. Each check uses the reference
that matches its question: the port against Python on identical pixels, the model against FP on identical pixels,
and the decoder effect is recorded. No threshold is loosened: the runtime check goes from 0.99 to 0.9999, and the
task-level gates are s4's. Accepting it means editing `gates.toml [s5_swift]`, `stages/s5_swift/` (compare, run,
README "known failure modes") and egcm-parity, in one commit with this ADR marked accepted.

## Reproduce

- Task level on the s5 vectors: `uv run --exact --group convert python -m scripts.s5_task_metrics`.
- The per-id, sensitivity and swapped-reference runs used throwaway scripts outside the repo (2026-10-25). The
  method is the one described under "How the pixels were made"; nothing in `stages/` or `gates.toml` was changed.
