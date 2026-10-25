import Foundation
import Synchronization
import Testing
@testable import EmbeddingGemma

/// Counts fetches and forwards them to the real URLSession fetcher (file:// URLs here), optionally slowly.
final class CountingFetcher: FileFetcher {
  let count = Mutex(0)
  let delay: Duration
  init(delay: Duration = .zero) { self.delay = delay }
  func fetch(_ url: URL, to destination: URL, resumeFile: URL, progress: (@Sendable (Int64) -> Void)?) async throws {
    count.withLock { $0 += 1 }
    if delay > .zero { try await Task.sleep(for: delay) }
    try await URLSessionFetcher().fetch(url, to: destination, resumeFile: resumeFile, progress: progress)
  }
}

/// A compiler stand-in: makes a folder with a marker file and records the package's files, or fails.
final class FakeCompiler: Sendable {
  enum Mode { case ok, fails, diskFull }
  let mode: Mode
  let calls = Mutex(0)
  let lastListing = Mutex<[String]>([])
  init(_ mode: Mode = .ok) { self.mode = mode }

  var compile: @Sendable (URL) async throws -> URL {
    { pkg in
      self.calls.withLock { $0 += 1 }
      self.lastListing.withLock { $0 = (try? FilesManifest.make(root: pkg).map(\.path)) ?? [] }
      switch self.mode {
      case .fails: throw CocoaError(.fileReadCorruptFile)
      case .diskFull: throw CocoaError(.fileWriteOutOfSpace)
      case .ok: break
      }
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

func store(_ root: URL = TestSupport.tempDir(), fetcher: CountingFetcher = CountingFetcher(),
           compiler: FakeCompiler = FakeCompiler(), capacity: Int64? = nil) -> ModelStore {
  ModelStore(root: root, fetcher: fetcher, compiler: compiler.compile, availableCapacity: { _ in capacity })
}

func expectError(_ body: () async throws -> Void, _ check: (EmbeddingGemmaError) -> Bool,
                 sourceLocation: SourceLocation = #_sourceLocation) async {
  do {
    try await body()
    Issue.record("no error thrown", sourceLocation: sourceLocation)
  } catch let e as EmbeddingGemmaError {
    #expect(check(e), "got \(e)", sourceLocation: sourceLocation)
  } catch {
    Issue.record("got \(error)", sourceLocation: sourceLocation)
  }
}

@Suite struct ModelStoreTests {
  @Test func manifestListsEveryFileWithSizeAndHash() throws {
    let root = try publishedFolder()
    let entries = try FilesManifest.decode(Data(contentsOf: root.appending(path: FilesManifest.fileName)))
    #expect(entries.map(\.path) == entries.map(\.path).sorted())
    #expect(entries.count == 5 && !entries.contains { $0.path == FilesManifest.fileName })
    let weight = try #require(entries.first { $0.path.hasSuffix("weight.bin") })
    #expect(weight.size == 100_000)
    #expect(weight.sha256 == (try Checksums.sha256(file: root.appending(path: weight.path))))
  }

  @Test func sha256MatchesKnownVector() throws {
    let f = TestSupport.tempDir().appending(path: "abc")
    try Data("abc".utf8).write(to: f)
    #expect(try Checksums.sha256(file: f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    #expect(Checksums.sha256(data: Data("abc".utf8)) == (try Checksums.sha256(file: f)))
  }

  static func manifest(_ paths: [String]) -> Data {
    let rows = paths.map { "{\"path\": \"\($0)\", \"size\": 1, \"sha256\": \"\(String(repeating: "0", count: 64))\"}" }
    return Data("[\(rows.joined(separator: ","))]".utf8)
  }

  @Test(arguments: ["../evil", "/abs/path", "a/../../b", "", "a//b", "./a"])
  func rejectsUnsafeManifestPaths(path: String) {
    #expect(throws: ManifestError.unsafePath(path)) { try FilesManifest.decode(Self.manifest([path])) }
  }

  @Test(arguments: ["files.json", ".egcm-install.json", "pkg/.lock", "a/b.download", "a/b.resume", ".DS_Store"])
  func rejectsStoreInternalNames(path: String) {
    #expect(throws: ManifestError.reservedName(path)) { try FilesManifest.decode(Self.manifest([path])) }
  }

  @Test func rejectsDuplicatePaths() {
    #expect(throws: ManifestError.duplicatePath("a/b")) { try FilesManifest.decode(Self.manifest(["a/b", "c", "a/b"])) }
  }

  @Test func installsVerifiesAndCompiles() async throws {
    let src = try publishedFolder(), fetcher = CountingFetcher(), s = store(fetcher: fetcher)
    let progress = Mutex<[InstallProgress]>([])
    let m = try await s.ensureModel(baseURL: src, revision: "r1") { p in progress.withLock { $0.append(p) } }
    #expect(m.revision == "r1" && m.repo == "local")
    #expect(m.folder == s.root.appending(path: "local/r1"))
    #expect(FileManager.default.fileExists(atPath: m.compiledModel.appending(path: "compiled-from").path))
    #expect(FileManager.default.fileExists(atPath: m.tokenizer.appending(path: "tokenizer.json").path))
    #expect(!FileManager.default.fileExists(atPath: m.folder.appending(path: "files").path))  // package removed
    #expect(InstallMarker.read(m.folder)?.complete == true)
    #expect(fetcher.count.withLock { $0 } == 6)  // files.json + 5 files
    let seen = progress.withLock { $0 }
    #expect(seen.last?.fraction == 1 && seen.last?.completedBytes == seen.last?.totalBytes)
    #expect(zip(seen, seen.dropFirst()).allSatisfy { $0.completedBytes <= $1.completedBytes })  // byte-level, monotonic
    #expect(try s.root.resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup == true)
    let again = try await s.ensureModel(baseURL: src, revision: "r1")  // installed: downloads nothing
    #expect(again == m && fetcher.count.withLock { $0 } == 6)
  }

  @Test func tamperedHashThrowsChecksumMismatch() async throws {
    let src = try publishedFolder()
    let tampered = "\(ModelStore.packageName)/Data/com.apple.CoreML/weights/weight.bin"
    try Data("tampered".utf8).write(to: src.appending(path: tampered))  // after files.json was written
    let s = store()
    await expectError({ _ = try await s.ensureModel(baseURL: src, revision: "r1") }) {
      if case .checksumMismatch(tampered) = $0 { true } else { false }
    }
    #expect(s.installed(repo: "local", revision: "r1") == nil)
    let leftovers = FileManager.default.enumerator(atPath: s.root.path)?.allObjects as? [String] ?? []
    #expect(!leftovers.contains { $0.hasSuffix("weight.bin") || $0.hasSuffix("weight.bin.download") })
  }

  @Test func sameSizeTamperIsCaughtByHash() async throws {
    let src = try publishedFolder()
    try Data("[]".utf8).write(to: src.appending(path: "tokenizer/tokenizer_config.json"))  // same 2 bytes as "{}"
    await expectError({ _ = try await store().ensureModel(baseURL: src, revision: "r1") }) {
      if case .checksumMismatch("tokenizer/tokenizer_config.json") = $0 { true } else { false }
    }
  }

  @Test func pinnedManifestHashIsEnforced() async throws {
    let src = try publishedFolder()
    let good = try Checksums.sha256(file: src.appending(path: FilesManifest.fileName))
    await expectError({
      _ = try await store().ensureModel(baseURL: src, revision: "r1", filesSHA256: String(repeating: "a", count: 64))
    }) { if case .checksumMismatch("files.json") = $0 { true } else { false } }
    _ = try await store().ensureModel(baseURL: src, revision: "r1", filesSHA256: good)
  }

  @Test func newRevisionsNeverDeleteOldOnesAndCleanupIsExplicit() async throws {
    let src = try publishedFolder(), root = TestSupport.tempDir(), fetcher = CountingFetcher()
    let r1 = try await store(root, fetcher: fetcher).ensureModel(baseURL: src, revision: "r1")
    await expectError({ _ = try await store(root, fetcher: fetcher, compiler: FakeCompiler(.fails))
      .ensureModel(baseURL: src, revision: "r2") }) { if case .modelLoadFailed = $0 { true } else { false } }
    #expect(store(root).installed(repo: "local", revision: "r1") == r1)

    // The retry reuses the verified files of the failed attempt (only files.json is fetched again) and deletes it.
    let before = fetcher.count.withLock { $0 }
    let r2 = try await store(root, fetcher: fetcher).ensureModel(baseURL: src, revision: "r2")
    #expect(fetcher.count.withLock { $0 } == before + 1)
    #expect(try FileManager.default.contentsOfDirectory(atPath: root.appending(path: "local").path).sorted() == ["r1", "r2"])

    let removed = try await store(root).removeRevisions(except: [r2])
    #expect(removed.map(\.lastPathComponent) == ["r1"])
    #expect(store(root).installed(repo: "local", revision: "r2") == r2)
  }

  @Test func sharedRootKeepsForeignFilesAndLoadedRevisions() async throws {
    let root = TestSupport.tempDir(), fm = FileManager.default
    try Data("mine".utf8).write(to: root.appending(path: "notes.txt"))
    try fm.createDirectory(at: root.appending(path: "local/not-ours"), withIntermediateDirectories: true)
    try fm.createDirectory(at: root.appending(path: "Other"), withIntermediateDirectories: true)
    let src = try publishedFolder()
    let r1 = try await store(root).ensureModel(baseURL: src, revision: "r1")
    let r2 = try await store(root).ensureModel(baseURL: src, revision: "r2")
    var lease: RevisionLease? = RevisionLease(r1.folder)  // r1 is loaded by a live EmbeddingGemma
    #expect(try await store(root).removeRevisions().isEmpty == false)
    #expect(store(root).installed(repo: "local", revision: "r1") == r1)  // in use: kept
    #expect(store(root).installed(repo: "local", revision: "r2") == nil)  // not in use, not kept: removed
    _ = lease
    lease = nil
    #expect(try await store(root).removeRevisions().map(\.lastPathComponent) == ["r1"])
    for path in ["notes.txt", "local/not-ours", "Other"] { #expect(fm.fileExists(atPath: root.appending(path: path).path)) }
  }

  @Test func concurrentCallsForOneRevisionShareOneInstall() async throws {
    let src = try publishedFolder(), fetcher = CountingFetcher(delay: .milliseconds(30)), compiler = FakeCompiler()
    let s = store(fetcher: fetcher, compiler: compiler)
    async let a = s.ensureModel(baseURL: src, revision: "r1")
    async let b = s.ensureModel(baseURL: src, revision: "r1")
    let (ma, mb) = try await (a, b)
    #expect(ma == mb)
    #expect(fetcher.count.withLock { $0 } == 6 && compiler.calls.withLock { $0 } == 1)
  }

  @Test func concurrentInstallsOfTwoRevisionsBothSucceed() async throws {
    let src1 = try publishedFolder(), src2 = try publishedFolder(extra: ["tokenizer/tokenizer.json": "{\"v\": 2}"])
    let s = store(fetcher: CountingFetcher(delay: .milliseconds(20)))
    async let a = s.ensureModel(baseURL: src1, revision: "r1")
    async let b = s.ensureModel(baseURL: src2, revision: "r2")
    let (m1, m2) = try await (a, b)
    #expect(s.installed(repo: "local", revision: "r1") == m1 && s.installed(repo: "local", revision: "r2") == m2)
    #expect(try String(contentsOf: m2.tokenizer.appending(path: "tokenizer.json"), encoding: .utf8) == "{\"v\": 2}")
  }

  @Test func unlistedFilesAreRemovedBeforeCompiling() async throws {
    let src = try publishedFolder(), root = TestSupport.tempDir()
    // A dead attempt left an extra file in its package folder.
    let stale = root.appending(path: "local/r1.partial-dead")
    let extra = stale.appending(path: "files/\(ModelStore.packageName)/Data/stale.bin")
    try FileManager.default.createDirectory(at: extra.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data("x".utf8).write(to: extra)
    try InstallMarker(repo: "local", revision: "r1", filesSHA256: nil, complete: false).write(stale)
    let compiler = FakeCompiler()
    _ = try await store(root, compiler: compiler).ensureModel(baseURL: src, revision: "r1")
    #expect(compiler.lastListing.withLock { $0 }.sorted() == ["Data/com.apple.CoreML/model.mlmodel",
                                                              "Data/com.apple.CoreML/weights/weight.bin", "Manifest.json"])
    #expect(!FileManager.default.fileExists(atPath: stale.path))
  }

  @Test func diskFullDuringCompileIsAStorageError() async throws {
    let src = try publishedFolder()
    await expectError({ _ = try await store(compiler: FakeCompiler(.diskFull)).ensureModel(baseURL: src, revision: "r1") }) {
      if case .storageFailed = $0 { true } else { false }
    }
  }

  @Test func insufficientStorageIsCheckedBeforeDownloading() async throws {
    let src = try publishedFolder(), fetcher = CountingFetcher()
    await expectError({ _ = try await store(fetcher: fetcher, capacity: 1000).ensureModel(baseURL: src, revision: "r1") }) {
      if case .insufficientStorage(let required, 1000) = $0 { required > 300_000 } else { false }
    }
    #expect(fetcher.count.withLock { $0 } == 1)  // only files.json
  }

  @Test func missingPackageOrTokenizerIsRejected() async throws {
    let src = try publishedFolder()
    try FileManager.default.removeItem(at: src.appending(path: "tokenizer"))
    try FilesManifest.write(root: src)
    await expectError({ _ = try await store().ensureModel(baseURL: src, revision: "r1") }) {
      if case .downloadFailed = $0 { true } else { false }
    }
  }

  @Test(arguments: ["", "../x", "a/b", ".", ".hidden", "r1.partial-x"])
  func unsafeRevisionsAreRejected(revision: String) async throws {
    let src = try publishedFolder()
    await expectError({ _ = try await store().ensureModel(baseURL: src, revision: revision) }) {
      if case .invalidSource = $0 { true } else { false }
    }
  }

  @Test(arguments: [("org/name", "main"), ("org/name", "pending"), ("org/name", String(repeating: "A", count: 40)),
                    ("name", String(repeating: "a", count: 40)), ("a/b/c", String(repeating: "a", count: 40)),
                    ("org/na me", String(repeating: "a", count: 40))])
  func huggingFaceSourcesAreValidated(repo: String, revision: String) async {
    await expectError({ _ = try await store().ensureModel(repo: repo, revision: revision) }) {
      if case .invalidSource = $0 { true } else { false }
    }
  }

  @Test func huggingFaceURLsArePinnedToTheRevision() throws {
    try ModelStore.validate(repo: "org/name.v2-x_y", revision: String(repeating: "0a", count: 20))
    let url = ModelStore.huggingFaceBase(repo: "org/name", revision: "abc123")
    #expect(url.absoluteString == "https://huggingface.co/org/name/resolve/abc123/")
    #expect(url.appending(path: "files.json").absoluteString == "https://huggingface.co/org/name/resolve/abc123/files.json")
  }

  @Test func defaultRootIsScopedByBundle() {
    #if os(macOS)
      #expect(ModelStore.defaultRoot.path.hasSuffix("/\(Bundle.main.bundleIdentifier ?? "egcm")/EmbeddingGemma"))
    #endif
  }
}
