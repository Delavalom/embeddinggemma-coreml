# s0_reference

Golden outputs from Google's original model, run the official way: `sentence-transformers` `encode`, FP32, CPU.
Every later stage is compared against these files.

Run: `just stage 0` (the `reference` uv group: `transformers==5.20.2`, `sentence-transformers>=6.1`). About 25 minutes
on a busy M1 Pro.

## Input

- `google/embeddinggemma-2` at `egcm.paths.HF_REVISION` (never `main`), downloaded into `artifacts/hf/` (1.4 GB).
  Loaded with `config_kwargs={"audio_config": None}`, `torch_dtype=float32`, 4 torch threads.
- `fixtures/` (texts, images, `MANIFEST.sha256`), written by `python -m stages.s0_reference.build_fixtures`.
  `just stage 0` refuses to run without the manifest.

## Outputs (`artifacts/s0_reference/`)

| File | Contents |
|---|---|
| `revision.txt` | the HF commit used |
| `keys.json` | every tensor in `model.safetensors`: `{key: [shape, dtype]}` |
| `texts.npz` | `order` (text ids), `emb` [200, 768] float32 (normalized), `ids_<id>` int32 token ids (prompt, bos and eos included) |
| `images_{70,280}.npz` | `order`, `bucket` (bucket name per image), `height`, `width`, `raw_soft_tokens`, `emb_raw` and `emb_bucket` [200, 768]; per image `pixel_values_<id>` (float32 [tokens·9, 768], padding dropped), `image_position_ids_<id>` (int32 (x, y)), `input_ids_<id>` (int32) |
| `debug.npz` | intermediates for t000 (`text_*`) and i000 at budget 70, bucket path (`image_*`); see below |
| `image_token_layout.json` | special token ids, the full `input_ids` and token strings for one fixture per bucket, and t000 |

- **`emb_raw`:** the processor resizes the original image itself (aspect-preserving, varying size).
- **`emb_bucket`:** we resize the uint8 image to the bucket size first (`torchvision` bicubic, antialias, as the processor
  does). The bucket is a fixed point of the processor's resize, so it does not resize again; the run asserts that the
  non-padding patch count is `bucket.tokens * 9`. `pixel_values`, `image_position_ids` and `input_ids` belong to this path.
- **Embedding path:** `model.encode({"image": img}, processing_kwargs={"image": {"max_soft_tokens": budget}},
  normalize_embeddings=True)`; texts use `prompt_name="query"` (kind `query`) or `"document"`. The run checks it
  against the manual path (processor → `EmbeddingGemma2Model.forward` → mean of `last_hidden_state` over
  `attention_mask` → L2 normalize) on t000 and two images at both budgets and requires cosine > 0.99999.
- **`debug.npz` keys:** `{text,image}_lm_inputs_embeds`, `_lm_per_layer_inputs` [1, S, 24, 512], `_lm_layer_{00,05,23}`
  (output of the decoder layer), `_lm_final_norm`, `_lm_last_hidden_state` (after `embedding_projection`, before pooling),
  `text_input_ids`; and for the image `image_vision_patch_embed` and `image_vision_layer_{00,15}` ([1, 630, 768],
  padding rows included), `image_vision_pooler` ([1, tokens, 768], after the `sqrt(768)` scale), `image_embed_vision`
  ([tokens, 512]), `image_input_ids`, `image_pixel_values`, `image_position_ids`.

## Gate (`[s0_reference]` in gates.toml)

- `deterministic`: the first 10 texts and the first 10 images (both budgets, both paths) are run again and every
  embedding, token id, patch and position array is bitwise identical.
- `license_apache`: the model card front matter says `license: apache-2.0`.
- `key_inventory_written`: `keys.json` is written and has `language_model.*`, `vision_tower.*` and `embed_vision.*` keys.

Also recorded, not gated: `path_agreement_min_cosine`, and in `details` the versions, key prefix counts, bucket counts,
token lengths and the largest difference between the processor's patches and a plain `uint8 / 255` patchify
(about 6e-8: the processor multiplies by `1/255` rather than dividing).

## Known failure modes

- **transformers changed:** the version is pinned in pyproject (`transformers==5.20.2`). If you bump it, re-vendor
  `vendor/reference/` from the same tag and diff the model code.
- **sentence-transformers `encode` dict input changed:** use the manual pooling path (`reference.manual_embedding`)
  and record that in `EMBEDDING_PATH`.
- **"processor resized a bucket image again":** the bucket table no longer matches `get_aspect_ratio_preserving_size`
  (for example the processor's budget kwarg was renamed and the default budget of 280 was used). Check that
  `processing_kwargs={"image": {"max_soft_tokens": ...}}` still reaches the image processor.
- **not deterministic:** usually a changed torch thread count or batch composition. Embeddings are computed one
  input at a time with `reference.THREADS` threads; keep it that way.
- **a fixture text is over 128 tokens:** the longest text function is `embed_text_128`, so every fixture text must fit.
  Shorten it in `fixtures/texts.jsonl` and rebuild the manifest.
- **out of memory:** the run keeps only embeddings and token ids in memory and streams the pixel arrays to disk
  (`reference.NpzWriter`). Close other heavy processes rather than batching.
