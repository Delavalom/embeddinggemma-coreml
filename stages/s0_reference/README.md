# s0_reference

Golden outputs from Google's original model, run the official way: `sentence-transformers` `encode`, FP32, CPU.
Every later stage is compared against these files.

Run: `just stage 0` (the `reference` uv group: `transformers==5.20.2`, `sentence-transformers>=6.1`). About 40 minutes
on a busy M1 Pro.

## Input

- `google/embeddinggemma-2` at `egcm.paths.HF_REVISION` (never `main`), downloaded into `artifacts/hf/` (1.4 GB).
  `model.safetensors` must have sha256 `egcm.paths.HF_MODEL_SHA256`. Loaded with `config_kwargs={"audio_config": None}`,
  `torch_dtype=float32`, 4 torch threads.
- `fixtures/` (texts, images, `MANIFEST.sha256`), written by `python -m stages.s0_reference.build_fixtures` (see
  `fixtures/README.md`). The run starts with `verify_manifest()` and raises on any mismatch.

## Outputs (`artifacts/s0_reference/`)

| File | Contents |
|---|---|
| `revision.txt` | the HF commit used |
| `keys.json` | every tensor in `model.safetensors`: `{key: [shape, dtype]}` |
| `texts.npz` | `order` (text ids), `emb` [200, 768] float32 (normalized), `ids_<id>` int32 token ids (prompt, bos and eos included) |
| `images_{70,280}.npz` | `order`, `bucket` (bucket name per image), `height`, `width`, `raw_soft_tokens` (the processor's `num_soft_tokens_per_image` on the original), `emb_raw` and `emb_bucket` [200, 768]; per image `pixels_<id>` (uint8 HxWx3, the bucket image), `image_position_ids_<id>` (int32 (x, y)), `input_ids_<id>` (int32) |
| `debug.npz` | intermediates for t000 (`text_*`) and i000 at budget 70, bucket path (`image_*`); see below |
| `image_token_layout.json` | special token ids, the full `input_ids` and token strings for one fixture per bucket, and t000 |

All `.npz` files are deflated and written one array at a time (`artifacts.NpzWriter`); a crash leaves no file.

- **`emb_bucket`** (the target for s1 to s5): we resize the uint8 image to the bucket size first (`torchvision` bicubic,
  antialias, as the processor does). The bucket is a fixed point of the processor's resize, so it does not resize
  again.
- **`emb_raw`** (diagnostic only): the processor resizes the original image itself (aspect-preserving, varying size).
  `details.raw_vs_bucket_cosine` gives min and median of `emb_raw · emb_bucket` per budget: the cost of bucketing.
- **Pixel contract (enforced, raises):** for every image and budget the processor's non-padding patches equal, bit
  for bit, `patchify(pixels_<id>)`, and its positions equal the bucket's row-major `(x, y)` meshgrid
  (`stages/s0_reference/pixels.py`). So `pixels_<id>` is a complete record of what the model saw. To turn it back
  into model input, use the one helper every downstream stage uses:

  ```python
  from egcm.fixtures import load_bucket_pixels
  x = load_bucket_pixels(budget, image_id)   # float32 [1, 3, H, W] == u8.astype(np.float32) * np.float32(1/255)
  ```

  Not `u8 / 255`: the processor multiplies by the float32 reciprocal, and division differs by up to 6e-8.
- **Colour:** fixture pixels are sRGB (Commons photos with ICC profiles were converted when the fixtures were built;
  no file carries a profile). The processor ignores profiles. So the Swift preprocessor (s5) must draw images into
  an sRGB context, or its pixels will not match.
- **Embedding path:** `model.encode({"image": img}, processing_kwargs={"image": {"max_soft_tokens": budget}},
  normalize_embeddings=True)`; texts use `prompt_name="query"` (kind `query`) or `"document"` (strict lookup).
- **Path agreement (enforced, raises):** before the main run, `encode` is checked against the manual path (processor
  → `EmbeddingGemma2Model.forward` → mean of `last_hidden_state` over `attention_mask` → L2 normalize) on the first
  query, the first document, the longest text, and the first two images at both budgets. Every cosine must be
  > 0.99999.
- **`debug.npz` keys:** `{text,image}_lm_inputs_embeds`, `_lm_per_layer_inputs` [1, S, 24, 512], `_lm_layer_{00,05,23}`
  (output of the decoder layer), `_lm_final_norm`, `_lm_last_hidden_state` (after `embedding_projection`, before pooling),
  `text_input_ids`; and for the image `image_vision_patch_embed` and `image_vision_layer_{00,15}` ([1, 630, 768],
  padding rows included), `image_vision_pooler` ([1, tokens, 768], after the `sqrt(768)` scale), `image_embed_vision`
  ([tokens, 512]), `image_input_ids`, `image_pixels` (uint8), `image_position_ids`.

## Gate (`[s0_reference]` in gates.toml)

- `deterministic`: an in-process rerun of a sample (the first query, the first document, the longest text, and the
  first image of each bucket shape at both budgets and on both paths) gives bitwise-identical embeddings, token ids,
  pixels, positions and soft-token counts. It is not a second process or a second machine.
- `license_apache`: the model card front matter says `license: apache-2.0`.
- `key_inventory_written`: `keys.json` is written and has `language_model.*`, `vision_tower.*` and `embed_vision.*` keys.

Also recorded, not gated: `path_agreement_min_cosine`, and in `details` the versions, key prefix counts, bucket counts,
the longest text, `raw_vs_bucket_cosine`, and the determinism sample.

## Known failure modes

- **"fixtures do not match MANIFEST.sha256":** a fixture file was edited by hand or not committed. Rebuild with
  `build_fixtures` (it refuses to change committed fixtures without `--accept-changes`).
- **"model.safetensors has sha256 …":** the download is corrupt or the revision changed. Delete `artifacts/hf/` and rerun.
- **transformers changed:** the version is pinned in pyproject (`transformers==5.20.2`). If you bump it, re-vendor
  `vendor/reference/` from the same tag and diff the model code.
- **sentence-transformers `encode` dict input changed:** use the manual pooling path (`reference.manual_embedding`)
  and record that in `EMBEDDING_PATH`.
- **"processor resized a bucket image again":** the bucket table no longer matches `get_aspect_ratio_preserving_size`
  (for example the processor's budget kwarg was renamed and the default budget of 280 was used). Check that
  `processing_kwargs={"image": {"max_soft_tokens": ...}}` still reaches the image processor.
- **"processor patches differ from patchify(u8)":** the processor's rescale or patch order changed. Read
  `image_processing_gemma4.py` in the installed transformers and update `pixels.py` and `egcm.fixtures.to_unit` together.
- **not deterministic:** usually a changed torch thread count or batch composition. Embeddings are computed one
  input at a time with `reference.THREADS` threads; keep it that way.
- **a fixture text is over 128 tokens:** the longest text function is `embed_text_128`, so every fixture text must fit.
  Shorten it in `fixtures/texts.jsonl` and rebuild the manifest.
- **out of memory:** the run keeps only embeddings and token ids in memory and streams the pixel arrays to disk.
  Close other heavy processes rather than batching.
- **build_fixtures: "downloaded bytes have sha256 …":** the file changed on Commons since it was pinned. Replace it in
  `commons_sources.json` (or `--requery`) and rebuild with `--accept-changes`.
