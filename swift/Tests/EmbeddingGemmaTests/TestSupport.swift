import CoreGraphics
import Foundation
@testable import EmbeddingGemma

/// Paths and shared downloads for the tests. Everything large lives in the repo's gitignored `artifacts/`.
enum TestSupport {
  /// Commit of google/embeddinggemma-2 every stage uses. Must equal `HF_REVISION` in egcm/paths.py
  /// (tests/test_s5_export.py checks it).
  static let hfRevision = "914f7f89142e33e77833254d9c9b90c3cef7303b"

  /// tokenizer.json is 32 MB, over the 20 MB limit for git, so it is downloaded once from the pinned revision and
  /// verified (the sha256 of tokenizer.json is its Git LFS oid on the Hub).
  static let tokenizerFiles = [
    ManifestEntry(path: "tokenizer.json", size: 32_170_510,
                  sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
    ManifestEntry(path: "tokenizer_config.json", size: 1_599,
                  sha256: "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874"),
  ]

  static let repoRoot = URL(filePath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
  static let fixturesDir = repoRoot.appending(path: "fixtures")
  static let testFixtures = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "Fixtures")
  static let s4Dir = repoRoot.appending(path: "artifacts/s4_compress")

  /// The s4 model package, if stage 4 has run (`EmbeddingGemma2.mlpackage` if present, else the first
  /// `.mlpackage`). End-to-end tests are skipped without it.
  static var s4Package: URL? {
    let items = (try? FileManager.default.contentsOfDirectory(at: s4Dir, includingPropertiesForKeys: nil)) ?? []
    let packages = items.filter { $0.pathExtension == "mlpackage" }.sorted { $0.path < $1.path }
    return packages.first { $0.lastPathComponent == ModelStore.packageName } ?? packages.first
  }

  /// The contract model of stages/s5_swift/make_contract_model.py, if it was built.
  static var contractPackage: URL? {
    let url = repoRoot.appending(path: "artifacts/s5_swift/contract/\(ModelStore.packageName)")
    return FileManager.default.fileExists(atPath: url.path) ? url : nil
  }

  /// A local published folder (the Hugging Face layout) for `package`: the package cloned in (APFS clones cost no
  /// space), the tokenizer under tokenizer/, and files.json.
  static func publish(package: URL) async throws -> URL {
    let root = tempDir().appending(path: "published")
    let tok = root.appending(path: ModelStore.tokenizerDir)
    try FileManager.default.createDirectory(at: tok, withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: package, to: root.appending(path: ModelStore.packageName))
    let cache = try await tokenizerFolder()
    for f in tokenizerFiles { try FileManager.default.copyItem(at: cache.appending(path: f.path), to: tok.appending(path: f.path)) }
    try FilesManifest.write(root: root)
    return root
  }

  /// The tokenizer folder (tokenizer.json + tokenizer_config.json), downloaded on first use.
  static func tokenizerFolder() async throws -> URL {
    let dir = repoRoot.appending(path: "artifacts/tokenizer/\(hfRevision)")
    let base = URL(string: "https://huggingface.co/google/embeddinggemma-2/resolve/\(hfRevision)/")!
    try await TokenizerCache.shared.ensure(dir: dir, base: base, files: tokenizerFiles)
    return dir
  }

  static func tempDir() -> URL {
    let url = FileManager.default.temporaryDirectory.appending(path: "egcm-tests-\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }
}

/// Serializes the one-time tokenizer download across parallel tests.
actor TokenizerCache {
  static let shared = TokenizerCache()
  private var done = Set<URL>()
  private var sharedEncoder: Task<TextEncoder, any Error>?

  /// One TextEncoder for all tests (loading the 32 MB tokenizer takes seconds in debug builds).
  func encoder() async throws -> TextEncoder {
    if sharedEncoder == nil { sharedEncoder = Task { try await TextEncoder(tokenizerFolder: TestSupport.tokenizerFolder()) } }
    return try await sharedEncoder!.value
  }

  func ensure(dir: URL, base: URL, files: [ManifestEntry]) async throws {
    if done.contains(dir) { return }
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    for entry in files {
      try await VerifiedDownload.fetch(entry, base: base, into: dir, fetcher: URLSessionFetcher())
    }
    done.insert(dir)
  }
}

struct TextFixture: Decodable { let id: String; let kind: String; let text: String; let ids: [Int32] }
struct TextFixtures: Decodable { let hf_revision: String; let texts: [TextFixture] }

struct PixelBudget: Decodable {
  let bucket: String; let height: Int; let width: Int; let sha256: String; let sample_b64: String
}
struct PixelImage: Decodable { let id: String; let file: String; let height: Int; let width: Int
  let budgets: [String: PixelBudget] }
struct SyntheticCase: Decodable { let height: Int; let width: Int; let budgets: [String: PixelBudget] }
struct PixelFixtures: Decodable {
  let hf_revision: String; let samples: Int; let images: [PixelImage]; let synthetic: [SyntheticCase]
}

extension TestSupport {
  static func load<T: Decodable>(_ type: T.Type, _ name: String) throws -> T {
    try JSONDecoder().decode(T.self, from: Data(contentsOf: testFixtures.appending(path: name)))
  }

  /// The procedural test image of `export_fixtures.synthetic`, bit for bit: gradients, hash noise and a
  /// checkerboard in 37x29 tiles, as an sRGB CGImage.
  static func synthetic(height: Int, width: Int) throws -> CGImage {
    var bytes = [UInt8](repeating: 255, count: width * height * 4)
    bytes.withUnsafeMutableBufferPointer { b in
      for y in 0..<height {
        for x in 0..<width {
          let (ux, uy) = (UInt32(x), UInt32(y))
          let region = (x / 37 + y / 29) % 3
          for c in 0..<3 {
            let v: UInt32 = switch region {
            case 0: UInt32((x + 2 * y + 60 * c) % 256)
            case 1: ((ux &* 73_856_093) ^ (uy &* 19_349_663) ^ UInt32(c * 83_492_791)) >> 24
            default: (x / 5 + y / 5) % 2 == 1 ? 255 : 0
            }
            b[(y * width + x) * 4 + c] = UInt8(v)
          }
        }
      }
    }
    let ctx = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    guard let image = ctx?.makeImage() else { throw EmbeddingGemmaError.invalidImage }
    return image
  }

  /// `(k * n) // samples` for k < samples, as in stages/s5_swift/export_fixtures.py.
  static func sampleIndices(_ n: Int, _ samples: Int) -> [Int] {
    n <= samples ? Array(0..<n) : (0..<samples).map { $0 * n / samples }
  }
}
