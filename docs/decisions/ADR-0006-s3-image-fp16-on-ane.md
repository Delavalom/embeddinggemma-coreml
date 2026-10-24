# ADR-0006: Two fixture images miss the s3 FP16 gate on the Neural Engine

**Status:** proposed, 2026-10-24. Not accepted: `gates.toml` is unchanged. The coordinator decides.

## Context

s3 (FP16, `CPU_AND_NE`, M1 Pro) at commit 8f5ec5d, after the fixes in that commit's history (FP16-safe pooling and
L2 norm, no concat on the sequence axis, per-function placement):

| | value | gate |
|---|---|---|
| `min_cosine_text` | 0.999965 (all 4 text functions ≥ 0.99996) | ≥ 0.995, pass |
| `min_ane_fraction` | 0.99864 (text: 5873/5881 ops; image: 16851/16853) | ≥ 0.95, pass |
| `min_cosine_image` | **0.99404** | ≥ 0.995, **fail** |

Per image function, minimum (mean) over its fixtures:

| function | n | min | mean |
|---|---|---|---|
| embed_image_70_tall | 45 | 0.99833 | 0.99974 |
| embed_image_70_portrait | 40 | 0.99914 | 0.99983 |
| embed_image_70_square | 10 | 0.99988 | 0.99995 |
| embed_image_70_landscape | 91 | 0.99942 | 0.99991 |
| embed_image_70_wide | 14 | **0.99404** (i141) | 0.99946 |
| embed_image_280_tall | 45 | 0.99967 | 0.99995 |
| embed_image_280_portrait | 40 | 0.99848 | 0.99988 |
| embed_image_280_square | 10 | 0.99982 | 0.99994 |
| embed_image_280_landscape | 91 | **0.99405** (i161) | 0.99986 |
| embed_image_280_wide | 14 | 0.99970 | 0.99994 |

398 of 400 image vectors pass; the next worst is 0.99833. Latency on this Mac (`CPU_AND_NE`, median of 20):
`embed_text_16` 6.5 ms, `embed_image_70_square` 47.8 ms.

### What was measured (i141 at embed_image_70_wide unless noted)

- **The same FP16 package on the GPU passes:** `CPU_AND_GPU` gives i141 0.99970 (and 0.99965 minimum over 8
  embed_image_70_tall images). The failure is specific to the Neural Engine.
- **The FP32 reference is fragile for these two images.** Rounding only the input pixels to FP16, everything else
  FP32: i161 0.99910, i141 0.99994, i032 1.00000.
- **It is not one op or one layer.** Neural Engine for the patch embedding only, FP32 after: 0.99938. Neural Engine for
  the patch embedding and vision layer 0, FP32 after: 0.99426. FP32 for the patch embedding and layer 0, Neural Engine
  after: 0.99415. Each vision layer alone on the Neural Engine with an exact FP32 input is accurate (token cosine
  ≥ 0.99997); the error of any part is amplified by the rest for these images (random FP32 noise of the same size is
  not: 0.99999).
- **Keeping one op class in FP32** (`FP16ComputePrecision(op_selector=...)`, vision layers 0-7): softmax 0.99419,
  attention matmuls 0.99416, norms 0.99433, GELU 0.99411, MLP down conv 0.99419, gate/up convs 0.99423, q/k/v/o convs
  0.99427, patch conv 0.99412. All convs in FP32 gives 0.99996, but that moves almost the whole tower to the CPU.
- **Exact rewrites that did not help:** a per-layer constant offset on the one massive residual channel (175, about
  -600) 0.99427; `x*x` instead of `pow(x, 2)` in the norms 0.99422 (i161 0.99349).

## Options

1. **GPU for the image functions, Neural Engine for text.** The multifunction package is unchanged; the Swift runtime
   loads `embed_image_*` with `.cpuAndGPU` and `embed_text_*` with `.cpuAndNeuralEngine` (one `MLModelConfiguration`
   per function). Measured: i141 0.99970 on the GPU; to be measured on all 400 images before acceptance. s3 would then
   measure images on `CPU_AND_GPU` and gate `min_ane_fraction` on the text functions only. This is the plan's
   pre-approved GPU fallback, applied to the image functions only; it changes what the s3 metrics mean, so it needs
   this ADR accepted and the `gates.toml` / s3 README change in the same commit.
2. **Keep everything on the Neural Engine and change the image gate** (for example 0.99 for the minimum, or keep 0.995
   on the 1st percentile). Measured: min 0.99404, every function's mean ≥ 0.99946. A relaxation; not proposed.
3. **More FP32 on the CPU** (whole early layers or all convs): fails the ≥ 0.95 Neural Engine share or loses most of
   the speed; not proposed.

## Proposal

Option 1, after measuring all 400 images and the latency of `embed_image_70_square` on `CPU_AND_GPU` (s6 records the
on-device speed against the 60 ms target).
