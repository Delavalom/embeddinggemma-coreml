// swift-tools-version: 6.0
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
    .target(name: "EmbeddingGemma", dependencies: [.product(name: "Tokenizers", package: "swift-transformers")]),
    .target(name: "EmbeddingIndex"),
    .executableTarget(name: "egcm-parity", dependencies: ["EmbeddingGemma"]),
    .testTarget(name: "EmbeddingGemmaTests", dependencies: ["EmbeddingGemma"], exclude: ["Fixtures"]),
    .testTarget(name: "EmbeddingIndexTests", dependencies: ["EmbeddingIndex"]),
    .testTarget(name: "EgcmParityTests", dependencies: ["egcm-parity"]),
  ]
)
