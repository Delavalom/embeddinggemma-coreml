# ADR-0005: Colour spaces are converted to sRGB; alpha is dropped keeping colours

**Status:** proposed, 2026-10-23. Measure in s5/s6, then accept (founder).

## Context

The reference processor (PIL `convert("RGB")`, then the resize) ignores ICC profiles and drops alpha. Our fixtures are
all sRGB and opaque, so s0 cannot tell us what the model expects for anything else. iPhone photos are usually
Display P3 HEIC, and screenshots or stickers can carry alpha.

## Decision

- **Colour:** the Swift preprocessor draws every image into an 8-bit sRGB context, so Display P3, Adobe RGB, gray
  and other colour spaces are *converted* to sRGB. PIL would read the raw P3 numbers as if they were sRGB. Converting
  is what the model saw in training (web images, overwhelmingly sRGB), so it is the semantically right input, even
  though it differs from a naive PIL run on the same file.
- **Alpha:** PIL's `convert("RGB")` drops alpha and keeps the stored (unpremultiplied) colours.
  - Upright 8-bit sRGB images with non-premultiplied alpha (typical PNGs) are read directly from their bytes, which
    does exactly that.
  - Anything else with alpha is drawn premultiplied and then unpremultiplied (vImage): exact where alpha is 255,
    within 8-bit rounding where it is partial, and black where alpha is 0 (PIL would keep the hidden colour).
- Opaque images are always drawn, never read from their data provider. ImageIO decodes a JPEG differently depending
  on whether it is first drawn or first read (up to 73/255 apart), and the drawn decode is closer to PIL (p99 2/255
  vs 4/255 on the fixtures). For bitmaps already in memory, drawing sRGB into sRGB is an exact copy.

## To measure (s5/s6)

- Cosine between the Swift vectors of a few Display P3 HEIC photos and the reference vectors of the same photos
  converted to sRGB (`sips --matchTo sRGB`, then s0's path). Expected ≥ 0.99, like the other image fixtures.
- For contrast, the same photos with the P3 numbers assigned as sRGB (what plain PIL does), to show how much the
  choice matters.
- One PNG with partial alpha against PIL's `convert("RGB")`.

## Consequences

- Swift and the reference can differ on P3 inputs by design. The comparison that matters is against the
  sRGB-converted reference.
- If the measurement shows P3-as-sRGB matching the reference better *and* retrieval does not suffer, revisit.
