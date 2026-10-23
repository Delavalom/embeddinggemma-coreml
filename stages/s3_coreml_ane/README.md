# s3_coreml_ane

Every entry point converted at FP16 into one multifunction package, loaded with `CPU_AND_NE` on this Mac's Neural
Engine (M1 Pro), and compared with s0. Also measures where Core ML places each op, and the FP32 rewrite's activation
ranges (the probe).

Run: `just stage 3` (the `convert` group). Needs fresh passing s0, s1, s2. About 1.5 hours: the probe, 14 FP16
conversions, a Neural Engine compile per function on first load, 600 predictions.

## What it does

1. **Probe** (`egcm/probe.py`, diagnostic, in its own process: `stages/s3_coreml_ane/probe_main.py`): the FP32 torch rewrite on all 200 texts and 2 images per bucket and
   budget. Max |x| per site: residual stream after every layer, embeddings, patch embedding, attention and MLP
   outputs, the MLP and PLE inner products, every RMSNorm input and output. Sites above 30,000 are listed as
   `overflow_risk` (FP16 max is 65,504). Full table in `artifacts/s3_coreml_ane/probe.json`.
2. Converts every function at `ct.precision.FLOAT16` with `egcm.parts.build` (one process per part and per merge,
   resumable; see the s2 README) (same contract as s2: int32 ids and mask, fp32 pixels, fp32
   `embedding`; Core ML adds the casts), merges, checks the contract, compiles.
3. **Neural Engine fraction** (`egcm/placement.py`): `MLComputePlan` of the compiled package with `CPU_AND_NE`; per
   function, the share of ops (those with a device; consts have none) whose preferred device is the Neural Engine.
   `min_ane_fraction` is the minimum over functions; every off-ANE op is listed per function.
4. Predicts every fixture with `CPU_AND_NE`; cosines against s0's vectors.
5. Latency (median and p90 of 20 calls after 3 warm-ups) of `embed_text_16` and `embed_image_70_square`.
6. Deletes the FP16 package (s4 converts again and compresses each part before merging, because coremltools 9 does
   not compress multifunction models). `EGCM_KEEP_PACKAGES=1` keeps it.

## Gate

`gates.toml [s3_coreml_ane]`: `min_cosine_text >= 0.995`, `min_cosine_image >= 0.995`, `min_ane_fraction >= 0.95`.

## FP16 design (already in the model)

- RMSNorm is the per-token dynamic form (`egcm/model/norm.py`): each token is divided by its own max |x| before
  squaring, so no norm overflows or loses eps to FP16 subnormals at any input scale.
- The pooler's `* sqrt(768)` is dropped (the one FP16 overflow upstream calls out); the q·k logit bound is checked
  at load (≤ 64; the checkpoint is 32.1); the mask is -1e4, finite in FP16.

## Known failure modes

- **Cosine fails, probe shows `overflow_risk` on the residual stream**: spec §7 mitigation (b), scale the embeddings
  and residual stream by 1/s and divide the post-norm weights by s (exact; see plan Task 14). Needs an ADR.
- **Cosine fails without overflow**: precision loss in a specific op; mitigation (c), keep those ops in FP32 with
  `op_selector`, then re-measure placement. Needs an ADR.
- **Neural Engine fraction below 0.95**: the fallback is a `CPU_AND_GPU` variant with FP32 norms, labelled in
  RESULTS.md and recorded in an ADR (pre-approved).
- **Expected off-ANE ops**: the text functions' table lookup prelude (`gather`, the int32 mask cast and the gather's
  index handling) and the fp32 input/output casts. These are a handful of ops per function.
- **First load is slow**: the Neural Engine compiler runs per function on first load (cached by the OS afterwards).
