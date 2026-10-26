---
license: apache-2.0
base_model: google/embeddinggemma-2
base_model_relation: quantized
library_name: coreml
pipeline_tag: feature-extraction
tags: [coreml, embeddings, multimodal-embedding, image-text, sentence-similarity, ios, macos, neural-engine]
---

<!-- Rendered by stages/s7_publish (card.py + model_card.md) from the stage reports. Do not edit on the Hub. -->

# EmbeddingGemma 2 for Core ML

This is [`$base_repo`](https://huggingface.co/$base_repo) (Google DeepMind's text and image embedding model, one
shared 768-d space) converted to **multifunction Core ML models that run on the Apple Neural Engine**, one per image
budget. Every stage of the conversion was checked against the original PyTorch model. It is an unofficial conversion. It is not made or
endorsed by Google.

- **Base model:** `$base_repo` at revision `$base_revision` (`model.safetensors` sha256 `$base_sha256`).
- **Modalities:** text and images. Audio and video are not converted.
- **Variant shipped:** `$variant`: $variant_description.
- **Layout:** one folder per image budget, `b70/` and `b280/`. Each holds the text functions and that budget's image
  functions; a device uses one budget only, so it downloads one folder's archive.
- **Size:** apps download one file, `b<budget>/$archive` (an LZFSE Apple Archive of the package and the tokenizer:
  $download_mb_70 MB for budget 70, $download_mb_280 MB for budget 280; the s4 gate is $max_download_mb MB for the
  larger), described by `b<budget>/$manifest` (size, SHA-256 and image budget). Unpacked it is $unpacked_mb_70 /
  $unpacked_mb_280 MB; the package alone is $package_mb_70 / $package_mb_280 MB. The same files are also in each
  folder unpacked, each listed with its size and SHA-256 in `b<budget>/files.json`.
- **Runs on:** iOS 18+ and macOS 15+ (the model uses `ios18` ops). It is built for `.cpuAndNeuralEngine`.
- **Swift package:** [`EmbeddingGemma`](https://github.com/$github) (download, verification,
  tokenizer, image preprocessing, plus an on-disk vector index). See "Use from Swift" below.

## License and terms

- **The weights in this repo are a modified version of `$base_repo`, under the Apache License, Version 2.0.**
  This is the license the upstream model card declares at the pinned revision (`license: $base_license`, linking
  Google's [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license), which is the Apache 2.0 text).
  The original model is by Google DeepMind. See `LICENSE` (the full Apache 2.0 text) and `NOTICE`.
- **Changes made (Apache 2.0, section 4(b)):** the original weights were re-implemented in PyTorch, converted to Core
  ML, cast to FP16, and compressed (int8 per-channel convolution weights, int4 per-block embedding table). The
  vectors are close to the original's but not identical (see "Accuracy").
- **Upstream use policy, passed through:** the upstream model card says *"Deployments must adhere to the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)."* That requirement comes from
  the upstream model, and it applies to this conversion too. The upstream card also says developers are responsible
  for application-level safeguards (retrieval filtering, fairness testing) suited to their use case. Read
  Google's [model card](https://huggingface.co/$base_repo) for its intended uses, limitations and ethical
  considerations. They all apply here.
- The tokenizer files (`tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`) are unmodified copies from
  the upstream revision above.
- The conversion code and the Swift package are Apache-2.0 too (the GitHub repo's `LICENSE` and `NOTICE`).

## Functions

One `.mlpackage` per image budget, `b70/EmbeddingGemma2.mlpackage` and `b280/EmbeddingGemma2.mlpackage`, each with
nine functions with fixed shapes: the four text functions (the same in both) and the five image functions of its
budget. Both were split from one 14-function package, and every function gives the same output bytes as there.
Every function returns `embedding`, float32 `[1, 768]`, L2-normalized.

| function | input | tokens |
|---|---|---|
| `embed_text_16`, `_32`, `_64`, `_128` | `input_ids` int32 `[1, L]`, `attention_mask` int32 `[1, L]` (1 = token, 0 = padding) | L = 16 / 32 / 64 / 128 |
| `embed_image_70_{tall,portrait,square,landscape,wide}` (in `b70/`) | `pixels` `[1, 3, H, W]`, RGB in 0-1 | 60-70 image tokens |
| `embed_image_280_{tall,portrait,square,landscape,wide}` (in `b280/`) | `pixels` `[1, 3, H, W]`, RGB in 0-1 | 256-280 image tokens |

Image shapes (H x W). The bucket comes from the aspect ratio r = width / height: tall r < 0.6, portrait r < 0.83,
square r <= 1.2, landscape r <= 1.67, wide above that.

| budget | tall | portrait | square | landscape | wide |
|---|---|---|---|---|---|
| 70 | 576 x 240 | 480 x 336 | 384 x 384 | 336 x 480 | 240 x 576 |
| 280 | 1152 x 528 | 960 x 672 | 768 x 768 | 672 x 960 | 528 x 1152 |

- **Text** is `<bos>` + tokens(prompt + text) + `<eos>`, padded with `<pad>` (id 0) to the smallest length that
  fits. Longer inputs are cut to 128 tokens, keeping `<bos>` and `<eos>` (the upstream model allows up to 8K
  tokens). Use the upstream prompts: `task: search result | query: ` for queries, `title: none | text: ` for
  documents (also `task: classification | query: ` and `task: clustering | query: `).
- **Images** are resized straight to the bucket size (no crop or padding). Use 8-bit sRGB, antialiased bicubic
  resampling, and scale by 1/255 with no mean or std normalization, as the reference processor does. The Swift
  package does this.
- **Matryoshka:** you can truncate the 768-d output to 512, 256 or 128 dims and re-normalize. The Swift package
  defaults to 256. All accuracy numbers below are at 768 dims.

## Accuracy against the original model

Measured by the pipeline on 200 hand-written texts (120 of them search queries) and 200 images (CC0 and public
domain photos from Wikimedia Commons, plus synthetic screenshots, memes and receipts; 14 categories), at both image
budgets. Cosine values are the **minimum** over all inputs, not the mean.

| stage | what is compared | min cosine, text | min cosine, image |
|---|---|---|---|
| s1 | our PyTorch rewrite (FP32) vs the official `sentence-transformers` model | $s1_text | $s1_image |
| s2 | Core ML FP32 vs the official model | $s2_text | $s2_image |
| s3 | Core ML FP16 on the Neural Engine vs the official model (at least $s3_ane of ops on the Neural Engine) | $s3_text | $s3_image (1st percentile $s3_p1_image) |
| s4 | **shipped `$variant`** on `CPU_AND_NE` vs the official model | **$text_cos** | **$image_cos** (1st percentile $p1_image_cos) |
| s5 | the Swift package end to end on a Mac (tokenizer + preprocessing + model) vs the official model on the same decoded pixels; vs Python Core ML on identical inputs: $s5_runtime | $s5_cos (text and image) | |

Retrieval with the shipped variant (the 120 queries ranking the 200 images, budget 70 / 280):

| metric | budget 70 | budget 280 |
|---|---|---|
| text-to-image recall@10 vs the FP reference on the same bucketed images | $recall_70 | $recall_280 |
| 5th-percentile Spearman of the top 50, same reference | $spearman_70 | $spearman_280 |
| text-to-image recall@10 vs the exact-aspect FP model (recorded, not gated) | $raw_recall_70 | $raw_recall_280 |

For $top1_in_top10 of queries (worst budget), the exact-aspect model's top-1 image is in the shipped model's top
10. Image-to-image precision@10 by category label is at most $i2i_drop below the exact-aspect FP model.
Swift tokenization matches the reference ids for every fixture text (`$s5_tokens`). The 99th-percentile pixel
difference against the reference preprocessing is $s5_pixel (0-1 units).

**Caveats:**

- **The retrieval gates use a bucketed FP reference.** The upstream processor resizes each image to its own
  aspect-preserving size. This model has five fixed shapes per budget, so even with no compression, the bucketed
  vector differs from the exact-aspect one (min cosine $raw_vs_bucket_70 at budget 70, $raw_vs_bucket_280 at 280).
  That is why recall@10 against the exact-aspect model is lower than recall@10 against the bucketed reference. The
  compression and FP16 gates (min cosine >= 0.99, recall@10 >= 0.95, Spearman >= 0.9) are therefore measured
  against the FP model on the same bucketed images. Bucketing gets its own task-level gates (top-1 in top 10
  >= 0.95, category precision@10 drop <= 0.03). Measured by category labels, bucketing costs no retrieval
  quality. Details: ADR-0007 in the GitHub repo.
- **Two images fall below the FP16 bar on the Neural Engine.** In FP16 on the Neural Engine, fixture images i141
  and i161 reach 0.99404 / 0.99405 cosine. The other 398 are at 0.998 or above. On the GPU these two images pass,
  but the GPU is 2.8x slower at budget 70. They change the top 10 of only a few queries: with only those two
  vectors swapped in, recall@10 stays at 0.9975 (budget 70) and 0.9992 (budget 280). They were accepted at task
  level, and images stay on the Neural Engine. The lowest image cosines of the shipped variant are $worst_images.
  Details: ADR-0006.
- The test set is small: 200 texts, 200 images. Multilingual text, code and long documents were not tested
  separately. For quality on standard benchmarks, see Google's model card. This conversion only claims closeness
  to the original.

## Speed and memory

**Mac** (s4, `CPU_AND_NE`, median of 20 runs after warm-up). These were measured on the build Mac; the v1.0.0
build Mac is an M1 Pro with macOS 26. `embed_text_16` takes $ms_text_16 ms and `embed_image_70_square` takes
$ms_image_70_square ms. In every function, at least $ane_fraction of the ops are placed on the Neural Engine (`MLComputePlan`, minimum
over the 14 functions, measured on the package the two budget packages were split from).

**iPhone** (s6, PhotoSearchDemo's benchmark: the newest 1,000 photos at budget 70, then 200 at budget 280, and 20
queries; times cover preprocessing + prediction):

| device | iOS | ms per image, 70 (median) | ms per image, 280 (median) | ms per query (median) | peak memory, MB |
|---|---|---|---|---|---|
$s6_devices

Worst device: $s6_ms_70 ms per image at 70, $s6_ms_280 ms at 280, $s6_ms_query ms per query, peak $s6_rss MB (the
gate is 800 MB; the target for budget 70 is 60 ms, which is recorded, not gated).

**Storage:** the Swift package downloads the archive of the device's image budget, `b<budget>/$archive`
($download_mb_70 MB at 70, $download_mb_280 MB at 280, resumable), checks its size and SHA-256 against
`b<budget>/$manifest`, unpacks it ($unpacked_mb_70 / $unpacked_mb_280 MB) with the AppleArchive framework and
compiles the package. It needs free space for the archive, the unpacked files and the compiled model while it
installs; afterwards it keeps the compiled model and the tokenizer. The s4 gate measures the larger archive,
$download_mb MB (limit $max_download_mb MB).

## Use from Swift

```swift
// Package.swift: .package(url: "https://github.com/$github", from: "1.0.0")
import EmbeddingGemma
import EmbeddingIndex

let model = try await EmbeddingGemma.load(.huggingFace(revision: EmbeddingGemma.defaultRevision))
let photo = try await model.embed(image: cgImage, orientation: .up)  // [Float], 256-d by default, normalized
let query = try await model.embed(query: "receipts from last week")
```

The package pins this repo (`$repo`) at a commit and the SHA-256 of each budget's `$manifest`, which pins its
archive's size and SHA-256. `budget: .auto` picks 280 tokens on A17 Pro and newer iPhones and on Macs, and 70 on
other devices, and loads that budget's folder (`b280/` or `b70/`) only.

## Use from Python (coremltools, macOS 15+)

```python
import numpy as np, coremltools as ct
from tokenizers import Tokenizer
from torchvision.transforms.functional import pil_to_tensor, resize, InterpolationMode
from PIL import Image

BUDGET = 280  # the folder you downloaded: b280/ or b70/; its text functions are the same in both
PKG = f"b{BUDGET}/EmbeddingGemma2.mlpackage"
tok = Tokenizer.from_file(f"b{BUDGET}/tokenizer/tokenizer.json")
# self-check: the layout the model was traced with is <bos> ... <eos>
assert tok.encode("task: search result | query: screenshots of tweets").ids == \
    [2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 75490, 529, 46331, 1]

def fn(name):
    return ct.models.MLModel(PKG, function_name=name, compute_units=ct.ComputeUnit.CPU_AND_NE)

def embed_text(text, prompt="task: search result | query: "):
    ids = tok.encode(prompt + text).ids
    ids = ids if len(ids) <= 128 else ids[:127] + [ids[-1]]
    n = next(L for L in (16, 32, 64, 128) if L >= len(ids))
    x = np.zeros((1, n), np.int32); x[0, :len(ids)] = ids
    m = np.zeros((1, n), np.int32); m[0, :len(ids)] = 1
    return fn(f"embed_text_{n}").predict({"input_ids": x, "attention_mask": m})["embedding"][0]

SHAPES = {70: {"tall": (576, 240), "portrait": (480, 336), "square": (384, 384), "landscape": (336, 480),
               "wide": (240, 576)},
          280: {"tall": (1152, 528), "portrait": (960, 672), "square": (768, 768), "landscape": (672, 960),
                "wide": (528, 1152)}}

def bucket(r):  # r = width / height
    return ("tall" if r < 0.6 else "portrait" if r < 0.83 else "square" if r <= 1.2
            else "landscape" if r <= 1.67 else "wide")

def embed_image(path):
    img = Image.open(path).convert("RGB")  # like the reference: 8-bit RGB, ICC profile ignored
    name = bucket(img.width / img.height)
    px = resize(pil_to_tensor(img), list(SHAPES[BUDGET][name]), InterpolationMode.BICUBIC, antialias=True)
    px = px.float() / 255
    return fn(f"embed_image_{BUDGET}_{name}").predict({"pixels": px[None].numpy()})["embedding"][0]

def truncate(v, dims=256):  # Matryoshka
    v = v[:dims]; return v / np.linalg.norm(v)
```

The model is compiled each time `MLModel` is created. To make many calls, keep one instance per function. The
Python path is not gated; the gated path is the Swift package (s5).

## Reproducibility

The model was built by an 8-stage pipeline in the GitHub repo. Each stage writes `artifacts/<stage>/report.json`,
which is checked against `gates.toml`. A stage counts only if its report passed and is current. Each report
records `git_sha` and `code_hash` (sha256 over the stage's code, the lock file and its gate table), and
`inputs_sha256` chains it to the reports it read. The reports behind this release:

| stage | status | git_sha | code_hash |
|---|---|---|---|
$provenance

Reference environment: transformers $transformers, sentence-transformers $sentence_transformers; conversion:
coremltools $coremltools. To rebuild: `just setup`, then `just stage 0` through `just stage 5` (`just stage 6` needs
an iPhone). Numbers measured on the CPU and the Neural Engine can
vary slightly from run to run.

$results

## Limitations

- Text and images only. Text is limited to 128 tokens; images are limited to budgets of 70 or 280 tokens and five
  fixed aspect buckets each.
- Results are close to the original model, not bit-identical (see "Accuracy"). Do not mix vectors from this model
  with vectors from the PyTorch model or another conversion in the same index. The Swift index stores the model
  revision for that reason.
- Intel Macs have no Neural Engine; they run on the CPU/GPU and are slower. The fastest settings are on the Neural
  Engine with `.cpuAndNeuralEngine`.
- The first load compiles the model on the device, which takes a while. The Swift package caches the compiled
  model.
- Everything in the upstream card's "Limitations" and "Ethical Considerations and Risks" applies.

## Citation

Please cite the original model as Google DeepMind's model card asks
([`$base_repo`](https://huggingface.co/$base_repo)). For this conversion:

```bibtex
@software{embeddinggemma_coreml,
  title  = {EmbeddingGemma 2 for Core ML},
  author = {Delavalom},
  year   = {2026},
  url    = {https://huggingface.co/$repo},
  note   = {Core ML conversion of google/embeddinggemma-2}
}
```
