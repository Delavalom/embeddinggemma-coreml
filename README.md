# EmbeddingGemma 2 for Core ML

This repo converts Google DeepMind's [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)
(text and image embeddings in one 768-d space) into a multifunction Core ML model that runs on the Apple Neural
Engine. Every stage of the conversion is checked against the original model. The repo also ships a Swift package
for using the model in iOS and macOS apps.

- **The model:** on Hugging Face at [`delavalom/embeddinggemma-2-coreml`](https://huggingface.co/delavalom/embeddinggemma-2-coreml).
  Its model card has the measured accuracy, speed, size, license terms and limitations.
- **The Swift package:** `EmbeddingGemma` (loading, download with checksum verification, tokenizer, image
  preprocessing) and `EmbeddingIndex` (an on-disk vector index). Supports iOS 18+ and macOS 15+. See
  [`docs/swift-package.md`](docs/swift-package.md).
- **The demo:** [`Examples/PhotoSearchDemo`](Examples/PhotoSearchDemo/README.md) indexes the newest 1,000 photos on
  an iPhone and searches them by text, fully on-device.
- **The numbers:** [`docs/RESULTS.md`](docs/RESULTS.md), generated from the stage reports.

This is an unofficial conversion. It is not made or endorsed by Google.

## Quick start (Swift)

```swift
// Package.swift
.package(url: "https://github.com/Delavalom/embeddinggemma-coreml", from: "1.0.0")
// target dependencies: .product(name: "EmbeddingGemma", package: "embeddinggemma-coreml")
```

```swift
import EmbeddingGemma

let model = try await EmbeddingGemma.load(.huggingFace(revision: EmbeddingGemma.defaultRevision))
let image = try await model.embed(image: cgImage)
let query = try await model.embed(query: "a dog on the beach")
let score = zip(image, query).reduce(0) { $0 + $1.0 * $1.1 }  // vectors are normalized: dot = cosine
```

The model is published as one package per image budget (ADR-0010), and a phone gets only its own: the first `load`
resolves the budget (`.auto`: 280 on A17 Pro and newer and on Macs, 70 otherwise) and downloads one file,
`b<budget>/EmbeddingGemma2.aar` (an LZFSE Apple Archive of that budget's model package and the tokenizer), checks
its size and SHA-256 against `b<budget>/manifest.json`, unpacks it and compiles the model once. The larger of the
two archives is measured by s4 and gated against `max_download_mb` in [`gates.toml`](gates.toml); the measured
number is in the model card and [`docs/RESULTS.md`](docs/RESULTS.md). An app that downloads and unpacks the
archive itself can hand the folder over with `.installed(folder, revision:)`, which moves it in place and needs
space only for the compiled model.

## What is in the model

The model has 14 fixed-shape functions, published as one `.mlpackage` per image budget (`b70/`, `b280/`), each with
the four text functions and that budget's five image functions (s4 builds and gates all 14 together, then splits
them and checks every function gives the same output bytes):

- `embed_text_{16,32,64,128}`: text up to 128 tokens (in both packages).
- `embed_image_{70,280}_{tall,portrait,square,landscape,wide}`: images at an image-token budget of 70 or 280, with
  five aspect-ratio buckets per budget.

The weights are FP16 with int8 per-channel convolution weights and an int4 per-block embedding table (`int8_table4`).
The model card on Hugging Face lists the input shapes, the preprocessing, and Python usage with coremltools.

## How it is built and checked

The pipeline has stages `s0` to `s7`. Each stage writes `artifacts/<stage>/report.json`, which is checked against
the thresholds in [`gates.toml`](gates.toml). Each report records `git_sha`, a `code_hash` of the stage's code and
gates, and an `inputs_sha256` chain over the reports it read. A report goes stale when any of these change.

| stage | what it does |
|---|---|
| s0_reference | runs the official `sentence-transformers` model (pinned revision, FP32, CPU) on 200 texts and 200 images; writes the golden vectors |
| s1_rewrite | a PyTorch rewrite built for Core ML, matched to s0 (min cosine >= 0.9999) |
| s2_coreml_fp32 | Core ML FP32, matched to s0 (>= 0.9999) |
| s3_coreml_ane | Core ML FP16 on the Neural Engine (text >= 0.995; images: min >= 0.99 and 1st percentile >= 0.995; recall@10 >= 0.95; >= 95% of ops on the Neural Engine) |
| s4_compress | compression variants, smallest first; the first one to pass every gate ships, split into one package per image budget (min cosine >= 0.99, recall@10 >= 0.95, and the device download, one LZFSE Apple Archive of a budget's package and the tokenizer, the larger of the two within `max_download_mb`) |
| s5_swift | the Swift package end to end on a Mac: identical token ids, pixels within 1/255, the same vectors as Python Core ML on identical inputs (min cosine >= 0.9999), and s4's task-level gates against the FP model on the same pixels |
| s6_device | PhotoSearchDemo's benchmark on a real iPhone (speed per image and per query, peak memory <= 800 MB) |
| s7_publish | uploads to Hugging Face, pins the Swift package to that commit, and tags the release (guarded) |

Decisions that changed a gate are recorded in [`docs/decisions/`](docs/decisions). ADR-0006 covers the two
fixture images that miss the FP16 bar on the Neural Engine and were accepted on retrieval recall. ADR-0007 covers
measuring compression against the bucketed FP reference, and the download size limit. ADR-0008 covers s5: the Swift
package is checked against Python Core ML and the FP model on its own decoded pixels, because ImageIO and PIL decode
JPEGs slightly differently and the model itself moves with that (recorded, not gated).

To reproduce (macOS with Xcode, [`uv`](https://docs.astral.sh/uv/) and [`just`](https://github.com/casey/just)):

```
just setup        # Python environment + Swift package resolve
just stage 0      # ... through `just stage 5`; `just stage 6` needs an iPhone in Developer Mode
just status       # what passed, what is stale
just results      # regenerate docs/RESULTS.md
just test         # pytest + swift test
```

The weights are downloaded from Hugging Face by s0. No weights or `.mlpackage` files are in git.
[`AGENTS.md`](AGENTS.md) covers the repo's rules, provenance and every command in detail.

## Repo layout

| path | what |
|---|---|
| `egcm/` | the PyTorch rewrite (`egcm/model/`), conversion, compression, metrics, reports and provenance |
| `stages/sN_*/` | one directory per stage: `run.py` plus a README with its gates and known failure modes |
| `swift/` | the Swift package (`EmbeddingGemma`, `EmbeddingIndex`, the `egcm-parity` CLI for s5) |
| `Examples/PhotoSearchDemo/` | the SwiftUI demo and the s6 benchmark runner |
| `fixtures/` | the 200 texts and 200 images (CC0 / public domain / synthetic; see `fixtures/README.md`) |
| `docs/` | decisions (ADRs), the design spec and plan, `RESULTS.md` |
| `vendor/reference/` | upstream Transformers code, kept for reading only (never imported) |

## License

The code is under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for third-party code.

The published Core ML model is a modified version of `google/embeddinggemma-2`. At the pinned revision, the
upstream model is licensed under Apache 2.0. Its model card also states that deployments must adhere to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), which applies to this conversion
too.

## Citation

Please cite the original model as its [model card](https://huggingface.co/google/embeddinggemma-2) asks. To cite
this conversion, use the BibTeX entry in the Hugging Face model card.
