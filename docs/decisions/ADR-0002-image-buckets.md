# ADR-0002: Fixed image buckets

**Status:** accepted, 2026-10-22

## Context

The upstream image processor resizes each image so it keeps its aspect ratio and fits a patch budget, with both sides a multiple of 48 (16-pixel patches x 3x3 pooling). The output shape therefore varies from image to image. Core ML on the Neural Engine needs fixed shapes, so one converted function cannot serve every aspect ratio.

## Decision

Use six fixed buckets, three aspect ratios for each of two token budgets. Each bucket size is a fixed point of the upstream resize formula (`egcm/buckets.py`, `aspect_preserving_size`), so the reference model and ours see identical pixels for an image that lands exactly on a bucket.

| Budget | Bucket | Size (H x W px) | Soft tokens |
|---|---|---|---|
| 70 | portrait | 480 x 336 | 10 x 7 = 70 |
| 70 | landscape | 336 x 480 | 7 x 10 = 70 |
| 70 | square | 384 x 384 | 8 x 8 = 64 |
| 280 | portrait | 960 x 672 | 20 x 14 = 280 |
| 280 | landscape | 672 x 960 | 14 x 20 = 280 |
| 280 | square | 768 x 768 | 16 x 16 = 256 |

A photo with aspect ratio `r = W/H` goes to portrait if `r < 0.83`, landscape if `r > 1.2`, and square otherwise (`choose_bucket`). Swift resizes the photo to the bucket size.

## Consequences

- An image is stretched to the nearest bucket, so the aspect ratio can be off by up to about 10% (more at the bucket boundaries, for example `r = 0.83` against 0.70 for the portrait bucket). The reference vector for that image used its exact aspect.
- The s4 recall gate (recall@10 against the reference vectors) measures this cost. If it fails because of bucketing, we add buckets in a new ADR before relaxing any gate.
- This supersedes the near-square 24 x 27 patch grid sketched in the design spec; the square buckets here are exact multiples of 48 on both sides.
