# s1_rewrite

Our plain-PyTorch rewrite (`egcm/model/`), loaded strictly from Google's weights (`egcm/weights.py`), run in FP32 on
the CPU on exactly the inputs s0 recorded, and compared with s0's golden vectors.

Run: `just stage 1` (the `convert` uv group). Needs a passing, fresh s0. About 30 to 50 minutes on a busy M1 Pro,
almost all of it the 280-token image buckets.

## Input

- `artifacts/s0_reference/`: `keys.json`, `texts.npz`, `images_{70,280}.npz`, `debug.npz`, `image_token_layout.json`.
  Bucket images are read only through `egcm.fixtures.load_bucket_pixels`.
- The HF snapshot at `egcm.config.HF_SNAPSHOT` (downloaded by s0): `config.json` and `model.safetensors`. Only the
  keys the rewrite uses are read, as FP32.

## What it does

1. Maps every HF key with `egcm.weights.RULES` and builds the text tower and vision encoder on the meta device, then
   `load_state_dict(strict=True, assign=True)`. Unused keys must all be audio (`audio_tower.*`, `embed_audio.*`).
   Checks the q·k logit bound (`head_dim * max|q_norm.w| * max|k_norm.w| <= 64`; the checkpoint measures 32.1).
2. Texts: each `ids_<id>` from `texts.npz` (prompt, bos and eos included) is padded with `pad_id` to the smallest of
   `TEXT_LENGTHS` = 16/32/64/128 and run through `TextEmbedder` for that length (padded keys masked with -1e4, mean
   pooling over the real tokens).
3. Images: one bucket at a time, an `ImageEmbedder` whose fixed prefix/suffix tokens come from
   `image_token_layout.json`; every image of that bucket at that budget. Each image's recorded `input_ids` must equal
   the bucket layout and its positions the bucket's row-major grid.
4. `debug.npz`: the same intermediates s0 hooked, recomputed by the rewrite (diagnostic only).

## Output

`artifacts/s1_rewrite/report.json`:

- `metrics`: `strict_load` (strict load succeeded and every unused key is audio), `min_cosine_text`,
  `min_cosine_image` (minimum over every fixture; images over both budgets).
- `details.text.by_length`, `details.image.by_bucket`: n, min, mean and the three worst fixtures per text length and
  per image function.
- `details.debug.{text,image}`: per tensor `max_abs`, `ref_max_abs` and `rel = max_abs / ref_max_abs`, in pipeline
  order. Upstream's vision rows are padded to budget·9 and its pooler output includes `* sqrt(768)`; the comparison
  takes the valid rows and puts the scale back.
- `details.keys` (counts, any non-audio unused key), `details.logit_bound_max`, versions, seconds.

## Gate

`gates.toml [s1_rewrite]`: `strict_load = true`, `min_cosine_text >= 0.9999`, `min_cosine_image >= 0.9999`.

## Known failure modes

Read `details.debug` first: the first tensor whose `rel` jumps above about 1e-4 is where the bug is.

- **Patch flatten order.** Upstream flattens each patch in (py, px, c) order; the conv weight is
  `input_proj.weight.view(C, 16, 16, 3).permute(0, 3, 1, 2)`. Shows up at `vision_patch_embed`.
- **Axial RoPE half order.** The x half comes first (`cat(fx, fx, fy, fy)`; position ids are (x, y)). Swapping or
  mirroring it leaves `vision_patch_embed` exact and breaks `vision_layer_00`.
- **layer_scalar.** Multiplies the whole residual stream after the PLE block, every layer. Missing it shows at
  `lm_layer_00` on both sides.
- **Mask sign.** Padded keys get `+(-1e4)`; a positive mask makes padded tokens dominate. Text only (images have no
  padding); compare `by_length` against the unpadded debug text.
- **PLE reshape order.** Upstream reshapes the projection to `[S, L, P]`: layer-major, so `split(P, dim=1)` is right.
  A dim-major split breaks `lm_per_layer_inputs`.
- **The pooler scale.** The rewrite drops `* sqrt(768)` (scale-invariant before the weightless RMSNorm). That is exact
  only while eps is negligible: the pooled mean square of the debug image is ≥ 175 against eps 1e-6.
- **Vision drift is larger than text.** FP32 summation-order noise at the patch embedding (about 1.5e-6 relative) is
  amplified about 20x by 16 vision layers (`vision_layer_15` about 3e-5). That is expected, not a bug; a bug gives
  `rel` of order 1e-2 or more.
