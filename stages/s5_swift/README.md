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
- `pixels.json`, `images`: 20 photos (4 per bucket shape; 2 of the tall ones are PNG screenshots) at both budgets:
  the bucket s0 chose, the sha256 of s0's full uint8 HxWx3 bucket image, and 8,192 values sampled from its uint8
  `[3, H, W]` tensor at indices `(k * 3*H*W) // 8192` (base64).
- `pixels.json`, `synthetic`: a procedural lossless pattern (reproduced exactly in Swift) at 10 sizes covering all
  ten buckets, resized by torch's uint8 antialiased bicubic (which reproduces s0 bit for bit), sampled the same way.

**The pixel gate is taken on decode-identical inputs** (`docs/decisions/ADR-0004-pixel-gate-inputs.md`): the
synthetic images and the PNG photos. There the 99th percentile of `|swift - reference|` (0-1 units) is gated at
`[s5_swift].max_pixel_abs_diff` (1/255), and in fact every case is bit-exact. JPEG photos decode differently in
ImageIO and PIL, so on them the number includes the decoder: it is printed and bounded at 3/255 for regressions only.

Measured 2026-10-23 on the M1 Pro:

| Resize | Inputs | p99 | max | bit-exact |
|---|---|---|---|---|
| vImage `vImageScale_ARGB8888`, high quality (tried first) | 20 photos x 2 budgets | 11/255 | 42/255 | 0/40 |
| torch uint8 AA bicubic port (`Resampler.swift`, Accelerate) | 10 synthetic x 2 budgets | 0 | 0 | 20/20 |
| same | 2 PNG photos x 2 budgets | 0 | 0 | 4/4 |
| same | 16 JPEG photos x 2 budgets (decoder included) | 2/255 | 31/255 | 0/32 |

Rerun the export whenever s0 is rerun; `tests/test_s5_export.py` checks the files are present, current (same
`HF_REVISION`), cover every bucket, and stay under 2 MB.

The tokenizer (`tokenizer.json`, 32 MB) is over the 20 MB limit for git. Tests download it once from
`google/embeddinggemma-2` at `HF_REVISION` into `artifacts/tokenizer/<revision>/` and verify its sha256; the app gets
it from the published model folder through `ModelStore`.

## Known failure modes

- **Pixel test fails after an s0 rerun:** the committed samples are stale. Rerun `export_fixtures`.
- **JPEG pixel numbers look bad but PNG and synthetic are exact:** that is the JPEG decoder (ImageIO vs PIL), not
  the resize. See ADR-0004. It shows up again, if anywhere, in the end-to-end cosine on JPEG fixtures.
- **Tokenizer download fails:** the tests need network once (huggingface.co). After that the verified cache in
  `artifacts/tokenizer/` is used.
