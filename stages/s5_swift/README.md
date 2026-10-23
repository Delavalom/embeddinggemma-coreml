# s5_swift

The Swift package on macOS against s0: token ids, preprocessed pixels, and (once s4 exists) end-to-end vectors.
`run.py` and the `egcm-parity` CLI land with Task 19; this README covers the unit-test fixtures for now.

## Unit-test fixtures

`swift test` checks the tokenizer and the image preprocessor against small committed files in
`swift/Tests/EmbeddingGemmaTests/Fixtures/`, written by:

```
uv run --exact --group convert python -m stages.s5_swift.export_fixtures [--s0 <artifacts/s0_reference>]
```

- `text_ids.json`: s0's token ids for all 200 fixture texts (prompt, `<bos>` and `<eos>` included).
- `pixels.json`: 20 images (4 per bucket shape; 2 of the tall ones are PNG screenshots) at both budgets: the
  bucket s0 chose, the sha256 of s0's full uint8 HxWx3 bucket image, and 8,192 values sampled from its uint8
  `[3, H, W]` tensor at indices `(k * 3*H*W) // 8192` (base64).

The pixel test computes the 99th percentile of `|swift - s0|` over the samples (0-1 units) and gates it at
`[s5_swift].max_pixel_abs_diff` (1/255). Rerun the export whenever s0 is rerun; `tests/test_s5_export.py` checks the
files are present, current (same `HF_REVISION`) and under 2 MB.

The tokenizer (`tokenizer.json`, 32 MB) is over the 20 MB limit for git. Tests download it once from
`google/embeddinggemma-2` at `HF_REVISION` into `artifacts/tokenizer/<revision>/` and verify its sha256; the app gets
it from the published model folder through `ModelStore`.

## Known failure modes

- **Pixel test fails after an s0 rerun:** the committed samples are stale. Rerun `export_fixtures`.
- **Tokenizer download fails:** the tests need network once (huggingface.co). After that the verified cache in
  `artifacts/tokenizer/` is used.
