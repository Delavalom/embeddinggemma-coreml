import Foundation
import Synchronization
import Testing
@testable import EmbeddingGemma

/// Counts fetches and forwards them to the real URLSession fetcher (file:// URLs here).
final class CountingFetcher: FileFetcher {
  let count = Mutex(0)
  func fetch(_ url: URL, to destination: URL, resumeFile: URL) async throws {
    count.withLock { $0 += 1 }
    try await URLSessionFetcher().fetch(url, to: destination, resumeFile: resumeFile)
  }
}

/// A compiler stand-in: makes a folder with a marker file, or fails.
struct FakeCompiler {
  var fails = false
  var compile: @Sendable (URL) async throws -> URL {
    let fails = fails
    return { pkg in
      if fails { throw CocoaError(.fileReadCorruptFile) }
      let out = TestSupport.tempDir().appending(path: "fake.mlmodelc")
      try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
      try Data(pkg.lastPathComponent.utf8).write(to: out.appending(path: "compiled-from"))
      return out
    }
  }
}

/// A published folder with the same layout as the Hugging Face repo: the package, the tokenizer, files.json.
func publishedFolder(extra: [String: String] = [:]) throws -> URL {
  let root = TestSupport.tempDir().appending(path: "published")
  let files = [
    "\(ModelStore.packageName)/Manifest.json": "{}",
    "\(ModelStore.packageName)/Data/com.apple.CoreML/model.mlmodel": "model bytes",
    "\(ModelStore.packageName)/Data/com.apple.CoreML/weights/weight.bin": String(repeating: "w", count: 100_000),
    "tokenizer/tokenizer.json": "{\"model\": {}}",
    "tokenizer/tokenizer_config.json": "{}",
  ].merging(extra) { $1 }
  for (path, text) in files {
    let url = root.appending(path: path)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(text.utf8).write(to: url)
  }
  try FilesManifest.write(root: root)
  return root
}

@Suite struct ModelStoreTests {
  @Test func manifestListsEveryFileWithSizeAndHash() throws {
    let root = try publishedFolder()
    let entries = try FilesManifest.decode(Data(contentsOf: root.appending(path: FilesManifest.fileName)))
    #expect(entries.map(\.path) == entries.map(\.path).sorted())
    #expect(entries.count == 5 && !entries.contains { $0.path == FilesManifest.fileName })
    let weight = try #require(entries.first { $0.path.hasSuffix("weight.bin") })
    #expect(weight.size == 100_000)
    #expect(weight.sha256 == "\(try Checksums.sha256(file: root.appending(path: weight.path)))")
  }

  @Test func sha256MatchesKnownVector() throws {
    let f = TestSupport.tempDir().appending(path: "abc")
    try Data("abc".utf8).write(to: f)
    #expect(try Checksums.sha256(file: f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
  }

  @Test(arguments: ["../evil", "/abs/path", "a/../../b", "", "a//b", "./a"])
  func rejectsUnsafeManifestPaths(path: String) {
    let json = Data("[{\"path\": \"\(path)\", \"size\": 1, \"sha256\": \"\(String(repeating: "0", count: 64))\"}]".utf8)
    #expect(throws: ManifestError.unsafePath(path)) { try FilesManifest.decode(json) }
  }

  @Test func installsVerifiesAndCompiles() async throws {
    let src = try publishedFolder(), fetcher = CountingFetcher()
    let store = ModelStore(root: TestSupport.tempDir(), fetcher: fetcher, compiler: FakeCompiler().compile)
    let progress = Mutex<[Double]>([])
    let m = try await store.ensureModel(baseURL: src, revision: "r1") { p in progress.withLock { $0.append(p) } }
    #expect(m.revision == "r1")
    #expect(m.compiledModel == store.root.appending(path: "r1/model.mlmodelc"))
    #expect(FileManager.default.fileExists(atPath: m.compiledModel.appending(path: "compiled-from").path))
    #expect(FileManager.default.fileExists(atPath: m.tokenizer.appending(path: "tokenizer.json").path))
    #expect(!FileManager.default.fileExists(atPath: store.root.appending(path: "r1/files").path))  // package removed
    #expect(fetcher.count.withLock { $0 } == 6)  // files.json + 5 files
    #expect(progress.withLock { $0.last } == 1.0)
    // Installed: a second call downloads nothing.
    let again = try await store.ensureModel(baseURL: src, revision: "r1")
    #expect(again == m && fetcher.count.withLock { $0 } == 6)
  }

  @Test func tamperedHashThrowsChecksumMismatch() async throws {
    let src = try publishedFolder()
    let tampered = "\(ModelStore.packageName)/Data/com.apple.CoreML/weights/weight.bin"
    try Data("tampered".utf8).write(to: src.appending(path: tampered))  // after files.json was written
    let store = ModelStore(root: TestSupport.tempDir(), fetcher: CountingFetcher(), compiler: FakeCompiler().compile)
    let error = await #expect(throws: EmbeddingGemmaError.self) {
      _ = try await store.ensureModel(baseURL: src, revision: "r1")
    }
    guard case .checksumMismatch(let file) = error else { Issue.record("got \(String(describing: error))"); return }
    #expect(file == tampered)
    #expect(store.installed(revision: "r1") == nil)
    #expect(!FileManager.default.fileExists(atPath: store.root.appending(path: "r1.partial/files/\(tampered)").path))
  }

  @Test func sameSizeTamperIsCaughtByHash() async throws {
    let src = try publishedFolder()
    let path = "tokenizer/tokenizer_config.json"
    try Data("[]".utf8).write(to: src.appending(path: path))  // same 2 bytes as "{}"
    let store = ModelStore(root: TestSupport.tempDir(), fetcher: CountingFetcher(), compiler: FakeCompiler().compile)
    let error = await #expect(throws: EmbeddingGemmaError.self) {
      _ = try await store.ensureModel(baseURL: src, revision: "r1")
    }
    guard case .checksumMismatch(let file) = error else { Issue.record("got \(String(describing: error))"); return }
    #expect(file == path)
  }

  @Test func keepsPreviousRevisionUntilTheNewCompileSucceeds() async throws {
    let src = try publishedFolder(), root = TestSupport.tempDir(), fetcher = CountingFetcher()
    let good = ModelStore(root: root, fetcher: fetcher, compiler: FakeCompiler().compile)
    let bad = ModelStore(root: root, fetcher: fetcher, compiler: FakeCompiler(fails: true).compile)
    let r1 = try await good.ensureModel(baseURL: src, revision: "r1")

    let error = await #expect(throws: EmbeddingGemmaError.self) { _ = try await bad.ensureModel(baseURL: src, revision: "r2") }
    guard case .modelLoadFailed = error else { Issue.record("got \(String(describing: error))"); return }
    #expect(good.installed(revision: "r1") == r1)
    #expect(good.installed(revision: "r2") == nil)

    // The retry reuses the verified files from the failed attempt: only files.json is fetched again.
    let before = fetcher.count.withLock { $0 }
    let r2 = try await good.ensureModel(baseURL: src, revision: "r2")
    #expect(fetcher.count.withLock { $0 } == before + 1)
    #expect(good.installed(revision: "r2") == r2)
    #expect(good.installed(revision: "r1") == nil)  // removed once r2 is in place
    #expect(try FileManager.default.contentsOfDirectory(atPath: root.path) == ["r2"])
  }

  @Test func missingPackageOrTokenizerIsRejected() async throws {
    let src = try publishedFolder()
    try FileManager.default.removeItem(at: src.appending(path: "tokenizer"))
    try FilesManifest.write(root: src)
    let store = ModelStore(root: TestSupport.tempDir(), fetcher: CountingFetcher(), compiler: FakeCompiler().compile)
    await #expect(throws: EmbeddingGemmaError.self) { _ = try await store.ensureModel(baseURL: src, revision: "r1") }
  }

  @Test(arguments: ["", "../x", "a/b", "."])
  func unsafeRevisionsAreRejected(revision: String) async throws {
    let store = ModelStore(root: TestSupport.tempDir(), fetcher: CountingFetcher(), compiler: FakeCompiler().compile)
    await #expect(throws: EmbeddingGemmaError.self) {
      _ = try await store.ensureModel(baseURL: try publishedFolder(), revision: revision)
    }
  }

  @Test func huggingFaceURLsArePinnedToTheRevision() {
    let url = ModelStore.huggingFaceBase(repo: "org/name", revision: "abc123")
    #expect(url.absoluteString == "https://huggingface.co/org/name/resolve/abc123/")
    #expect(URL(string: "files.json", relativeTo: url)!.absoluteString
            == "https://huggingface.co/org/name/resolve/abc123/files.json")
  }

  @Test func staleResumeDataFallsBackToAFullDownload() async throws {
    let src = TestSupport.tempDir().appending(path: "f.bin"), dir = TestSupport.tempDir()
    try Data("payload".utf8).write(to: src)
    let dest = dir.appending(path: "f.bin"), resume = dir.appending(path: "f.bin.resume")
    try Data("not resume data".utf8).write(to: resume)
    try await URLSessionFetcher().fetch(src, to: dest, resumeFile: resume)
    #expect(try Data(contentsOf: dest) == Data("payload".utf8))
    #expect(!FileManager.default.fileExists(atPath: resume.path))
  }
}
