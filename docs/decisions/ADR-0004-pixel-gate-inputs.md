# ADR-0004: The s5 pixel gate is measured on decode-identical inputs

**Status:** proposed, 2026-10-23 (needs the founder's sign-off; `gates.toml` is unchanged)

## Context

`[s5_swift] max_pixel_abs_diff = 0.003922` (1/255 at the 99th percentile) was meant to prove that the Swift
preprocessor reproduces the reference preprocessing (RGB, bicubic + antialias, rescale). The plan measured it
against s0's bucket pixels for fixture photos. Most of those photos are JPEGs, and s0 decodes them with PIL
(libjpeg-turbo), while the Swift package can only use ImageIO, the system decoder on iOS and macOS.

Measured on 2026-10-23 (Task 16):

- **The decoders disagree.** ImageIO vs PIL on the same 4:2:0 JPEG at full size: up to 48/255 per pixel, p99 6 to
  7/255, about 22 to 36% of values differ (chroma upsampling and IDCT differ; no systematic bias). PNGs decode
  identically.
- **vImage misses the gate even without that.** `vImageScale_ARGB8888` + `kvImageHighQualityResampling`: p99 11/255,
  max 42/255, 0 of 40 images bit-exact.
- **The fallback is bit-exact.** A port of torch's uint8 `_upsample_bicubic2d_aa` (int16 fixed-point weights,
  horizontal pass first, uint8 intermediate), run with Accelerate in exact Double arithmetic, gives the same bytes as
  torch on every lossless input: 10 synthetic images covering all ten buckets (downscale up to 6.75x, upscale, the
  identity case) and the 2 PNG fixtures at both budgets. torch 2.7's `interpolate(uint8, bicubic, antialias=True)`
  reproduces s0's stored pixels bit for bit, so it is a valid reference.
- **With JPEG decoding included**, the same port gives p99 2/255 (p99.9 4/255, max 31/255) on the 16 JPEG fixtures.
  That fails a 1/255 gate, and no resize can fix it: only a libjpeg-turbo-compatible JPEG decoder could, which is
  out of scope (and app photos arrive as HEIC or JPEG decoded by Apple anyway).

## Decision

- `max_pixel_abs_diff` (s5, and the Swift unit test) is the 99th percentile of `|swift - reference|` over
  **decode-identical inputs**: the synthetic images and the PNG fixtures, against torch's uint8 resize (synthetic)
  and s0's stored pixels (PNG). The threshold stays 1/255.
- The decode-inclusive number on the JPEG fixtures is recorded (s5 `details.jpeg_pixel_p99`, and printed by the
  unit test), with a regression bound of 3/255 in the unit test only. It is not a gate.
- What the JPEG decoder does to the vectors is covered by the end-to-end gate `min_cosine >= 0.99` against s0, which
  is computed from the same ImageIO-decoded JPEGs.

## Consequences

- `gates.toml` does not change; the metric's inputs do. This ADR exists because that changes what the gate means.
- If s5's end-to-end cosine fails on JPEG fixtures and passes on PNG ones, the decoder is the cause; the fix would be
  a different decoder, not a looser cosine gate.
- `stages/s5_swift/export_fixtures.py` exports the synthetic cases; `swift/Tests/EmbeddingGemmaTests/
  ImagePreprocessorTests.swift` holds the gate and the recorded JPEG numbers.
