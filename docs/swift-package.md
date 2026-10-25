# EmbeddingGemma (Swift package)

<!-- Kept in docs/, not swift/: swift/ is in the code_hash of s5-s7, so editing a README there would make their reports stale. -->

On-device EmbeddingGemma 2 for iOS and macOS: text and image embeddings in one vector space, computed by the Core
ML conversion of `google/embeddinggemma-2` on the Neural Engine.

Two libraries:

- **`EmbeddingGemma`** downloads the model from Hugging Face at a pinned commit: `manifest.json`, then the one
  archive it names, `EmbeddingGemma2.aar` (LZFSE Apple Archive), checked against its size and SHA-256 and unpacked
  with the AppleArchive framework. It compiles the model once, tokenizes text, preprocesses images exactly as the reference
  processor does, and returns normalized vectors.
- **`EmbeddingIndex`** is a flat, exact nearest-neighbour index on disk (dot-product scores, FP16 storage), tied to
  the model revision so vectors from different models are never mixed.

## Requirements

- iOS 18+ or macOS 15+ (the model uses `ios18` Core ML ops). The package uses Swift tools 6.0 (Xcode 16 or later).
- A device with a Neural Engine is recommended. Intel Macs run the model on the CPU/GPU, which is slower.
- One download on first use, `EmbeddingGemma2.aar` (the model package and the tokenizer). Its size is gated by s4
  (`max_download_mb` in `gates.toml`); the measured size is in the model card and `docs/RESULTS.md`, and
  `manifest.json` has the exact `size` and `unpacked_size`. While it installs it needs free space for the archive,
  the unpacked files and the compiled model; afterwards the compiled model and the tokenizer stay in Application
  Support. Resumable: an interrupted download continues where it stopped.
- Dependency: [`swift-transformers`](https://github.com/huggingface/swift-transformers) (`Tokenizers`), from
  1.3.5, up to the next minor version.

## Installation (Swift Package Manager)

In Xcode, choose **File > Add Package Dependencies...**, enter
`https://github.com/Delavalom/embeddinggemma-coreml`, and pick version `1.0.0` or later. Or in `Package.swift`:

```swift
dependencies: [
  .package(url: "https://github.com/Delavalom/embeddinggemma-coreml", from: "1.0.0"),
],
targets: [
  .target(name: "MyApp", dependencies: [
    .product(name: "EmbeddingGemma", package: "embeddinggemma-coreml"),
    .product(name: "EmbeddingIndex", package: "embeddinggemma-coreml"),
  ]),
]
```

Each package release pins one model: `EmbeddingGemma.defaultRepo`, `defaultRevision` (the Hugging Face commit),
and `defaultManifestSHA256` (the SHA-256 of `manifest.json`, which pins the archive's size and SHA-256). The release
process writes all three and prints them on one `PUBLISHED_MODEL {"repo", "revision", "manifest_sha256"}` line.

## Example: search photos by text

```swift
import EmbeddingGemma
import EmbeddingIndex

// Load: downloads and verifies on first run, then reuses the installed copy.
let model = try await EmbeddingGemma.load(
  .huggingFace(revision: EmbeddingGemma.defaultRevision),
  budget: .auto,        // 280 image tokens on A17 Pro and newer iPhones and on Macs, 70 otherwise
  dimensions: 256       // Matryoshka: 1...768, truncated and re-normalized
) { progress in print("model download", progress.fraction) }
try await model.prewarm()  // optional: load the functions now instead of on first use

// Index
let url = URL.applicationSupportDirectory.appending(path: "photos.index")
let index = try VectorIndex(url: url, dimensions: model.dimensions, modelRevision: model.modelRevision)
for (id, image) in photos {                       // id: String, image: CGImage
  try index.upsert(id: id, vector: try await model.embed(image: image, orientation: .up))
}
try index.save()

// Search
let query = try await model.embed(query: "receipts from a restaurant")
for hit in index.search(query, k: 10) { print(hit.id, hit.score) }
```

- `embed(query:)` adds the search-query prompt. `embed(text:task:)` takes `.document` (the default),
  `.classification` or `.clustering`. Texts are cut at 128 tokens.
- `embed(image:orientation:)` takes a `CGImage` plus its EXIF orientation. Photos often need it.
- Calls can run concurrently. `unloadFunctions()` frees the loaded functions, for example on a memory warning.
- `.file(URL)` loads a local folder with the plain layout (`files.json`, `EmbeddingGemma2.mlpackage/`,
  `tokenizer/`), or with `manifest.json` + the archive, and copies it into the store (it needs free space for the
  copy too). That is useful for bundling the model in an app; the demo app does this.
- `.installed(folder, repo:, revision:)` adopts a folder your app already downloaded, verified and unpacked
  (`EmbeddingGemma2.mlpackage/`, `tokenizer/`): it is **moved** into the store (same volume, no copy) and only the
  package is compiled, so it needs free space for the compiled model only, about the package's size. Use the
  published repo and commit as `repo` and `revision`, so a later `.huggingFace` load finds it installed.
- `ModelStore().removeRevisions(except:)` deletes old model revisions after an upgrade.
- Errors are `EmbeddingGemmaError` (a `LocalizedError`): download, checksum, storage, load, tokenizer and
  contract failures.

[`Examples/PhotoSearchDemo`](../Examples/PhotoSearchDemo/README.md) is a complete SwiftUI app built on these
APIs.

## Accuracy and speed

The package is gated end to end against the original model (stage s5: identical token ids, pixels within 1/255,
min cosine >= 0.99 over 600 vectors). Measured numbers, including iPhone speed and memory, are in the
[model card](https://huggingface.co/Delavalom/embeddinggemma-2-coreml) and in [`docs/RESULTS.md`](RESULTS.md).

## Development

```
cd swift && swift test   # the first run downloads the tokenizer (32 MB) into artifacts/tokenizer/
```

`egcm-parity` is the CLI that stage s5 runs. It is not part of the public API.

## License

Apache License 2.0 ([`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE)). The model this package downloads is a
modified version of `google/embeddinggemma-2`, which is under Apache 2.0. Its use must also follow the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
