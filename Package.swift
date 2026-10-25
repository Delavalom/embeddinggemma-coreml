// swift-tools-version: 6.0
// The Swift package as consumers resolve it: `.package(url: "https://github.com/Delavalom/embeddinggemma-coreml", ...)`.
// SwiftPM only reads a manifest at the repository root, so this one points at the sources in swift/. Development
// (tests, the egcm-parity CLI) uses swift/Package.swift. Keep both in step: same platforms, dependencies, targets.
// It lives outside swift/ so editing it never changes the s5-s7 code hashes.
import PackageDescription

let package = Package(
  name: "EmbeddingGemma",
  platforms: [.iOS(.v18), .macOS(.v15)],
  products: [
    .library(name: "EmbeddingGemma", targets: ["EmbeddingGemma"]),
    .library(name: "EmbeddingIndex", targets: ["EmbeddingIndex"]),
  ],
  dependencies: [.package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.3.5"))],
  targets: [
    .target(name: "EmbeddingGemma", dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
            path: "swift/Sources/EmbeddingGemma"),
    .target(name: "EmbeddingIndex", path: "swift/Sources/EmbeddingIndex"),
  ]
)
