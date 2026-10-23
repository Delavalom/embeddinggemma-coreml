# ADR-0002: Fixed image buckets

**Status:** accepted, 2026-10-22 (amended the same day: tall and wide buckets added after review)

## Context

The upstream image processor resizes each image so it keeps its aspect ratio and fits a patch budget, with both sides a multiple of 48 (16-pixel patches x 3x3 pooling). The output shape therefore varies from image to image. Core ML on the Neural Engine needs fixed shapes, so one converted function cannot serve every aspect ratio.

The first version had three buckets per budget (portrait, square, landscape). Phone screenshots (about 1:2.2) and panoramas would have been stretched by 1.4x to 2x, which is a real quality cost for the photo-cleanup use case, where screenshots are a main category.

## Decision

Use ten fixed buckets: five aspect ratios for each of two token budgets (70 and 280). Each bucket size is a fixed point of the upstream resize formula (`egcm/buckets.py`, `aspect_preserving_size`; the tests check this against the vendored upstream function), so the reference model and ours see identical pixels for an image that lands exactly on a bucket.

| Budget | Bucket | Size (H x W px) | Patch grid | Soft tokens | Function |
|---|---|---|---|---|---|
| 70 | tall | 576 x 240 | 36 x 15 | 12 x 5 = 60 | `embed_image_70_tall` |
| 70 | portrait | 480 x 336 | 30 x 21 | 10 x 7 = 70 | `embed_image_70_portrait` |
| 70 | square | 384 x 384 | 24 x 24 | 8 x 8 = 64 | `embed_image_70_square` |
| 70 | landscape | 336 x 480 | 21 x 30 | 7 x 10 = 70 | `embed_image_70_landscape` |
| 70 | wide | 240 x 576 | 15 x 36 | 5 x 12 = 60 | `embed_image_70_wide` |
| 280 | tall | 1152 x 528 | 72 x 33 | 24 x 11 = 264 | `embed_image_280_tall` |
| 280 | portrait | 960 x 672 | 60 x 42 | 20 x 14 = 280 | `embed_image_280_portrait` |
| 280 | square | 768 x 768 | 48 x 48 | 16 x 16 = 256 | `embed_image_280_square` |
| 280 | landscape | 672 x 960 | 42 x 60 | 14 x 20 = 280 | `embed_image_280_landscape` |
| 280 | wide | 528 x 1152 | 33 x 72 | 11 x 24 = 264 | `embed_image_280_wide` |

An image with aspect ratio `r = width / height` goes to:

| r | Bucket |
|---|---|
| r < 0.6 | tall |
| 0.6 <= r < 0.83 | portrait |
| 0.83 <= r <= 1.2 | square |
| 1.2 < r <= 1.67 | landscape |
| r > 1.67 | wide |

Examples: an iPhone screenshot (1170 x 2532, r = 0.46) is tall; 4:3 landscape is landscape; 16:9 (r = 1.78) is wide. Swift resizes the photo to the bucket size.

## Consequences

Stretch is `max(r / b, b / r)` where `b = width / height` of the bucket, taken at the ends of the range of `r` that maps to the bucket (`worst_case_stretch` in `egcm/buckets.py`):

| Bucket | b (70 / 280) | Worst-case stretch, 70 | Worst-case stretch, 280 |
|---|---|---|---|
| tall | 0.417 / 0.458 | 1.44 at r = 0.6; unbounded below | 1.31 at r = 0.6; unbounded below |
| portrait | 0.700 | 1.19 (at r = 0.83) | 1.19 (at r = 0.83) |
| square | 1.000 | 1.21 (at r = 0.83) | 1.21 (at r = 0.83) |
| landscape | 1.429 | 1.19 (at r = 1.2) | 1.19 (at r = 1.2) |
| wide | 2.400 / 2.182 | 1.44 at r = 1.67; unbounded above | 1.31 at r = 1.67; unbounded above |

- The earlier "about 10%" figure was wrong: the middle three buckets stretch up to about 21% at their edges, and tall and wide have no outer limit. Typical cases are much smaller: an iPhone screenshot is 1.11x, 9:21 is 1.03x, 16:9 is 1.35x at budget 70, and a 3:1 panorama is 1.25x.
- The reference vector for each image used its exact aspect, so the s4 recall gate (recall@10 against the reference vectors) measures what this costs. If it fails because of bucketing, we add buckets in a new ADR before relaxing any gate.
- Ten image functions go into the multifunction `.mlpackage` instead of six. Weights are shared, so the size cost is small, but s2 to s4 convert and test more functions.
- tall and wide use fewer tokens than the budget (60 and 264) because the side lengths must be multiples of 48.
