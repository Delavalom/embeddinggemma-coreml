import Foundation
import Testing
@testable import EmbeddingGemma

/// `ModelStore.adopt` / `.installed(folder)`: an unpacked, verified layout moved into the store, not copied.
@Suite struct AdoptTests {
  /// The unpacked layout on the store's volume (the temp dir), without files.json.
  static func unpacked() throws -> (URL, Int64) {
    let folder = try publishedFolder()
    try FileManager.default.removeItem(at: folder.appending(path: FilesManifest.fileName))
    let pkg = try FilesManifest.make(root: folder).filter { $0.path.hasPrefix(ModelStore.packageName + "/") }
    return (folder, pkg.reduce(0) { $0 + $1.size })
  }

  @Test func adoptsInPlaceWithoutDownloadingOrCopying() async throws {
    let (folder, _) = try Self.unpacked(), fetcher = CountingFetcher(), compiler = FakeCompiler()
    let s = store(fetcher: fetcher, compiler: compiler)
    let m = try await s.adopt(folder: folder, repo: "owner/name", revision: String(repeating: "c", count: 40))
    #expect(fetcher.count.withLock { $0 } == 0 && compiler.calls.withLock { $0 } == 1)
    #expect(!FileManager.default.fileExists(atPath: folder.path))  // moved, not copied
    #expect(FileManager.default.fileExists(atPath: m.tokenizer.appending(path: "tokenizer.json").path))
    #expect(try FileManager.default.contentsOfDirectory(atPath: m.folder.path).sorted()
            == [".egcm-install.json", ModelStore.compiledName, ModelStore.tokenizerDir])  // package deleted
    #expect(s.installed(repo: "owner/name", revision: String(repeating: "c", count: 40)) == m)
  }

  @Test func needsSpaceForTheCompiledModelOnly() async throws {
    let (folder, packageBytes) = try Self.unpacked()
    await expectError({ _ = try await store(capacity: packageBytes - 1).adopt(folder: folder, repo: "local", revision: "r1") }) {
      if case .insufficientStorage(let required, _) = $0 { required == packageBytes } else { false }
    }
    #expect(FileManager.default.fileExists(atPath: folder.appending(path: ModelStore.packageName).path))  // untouched
    _ = try await store(capacity: packageBytes).adopt(folder: folder, repo: "local", revision: "r1")  // not 3x
  }

  @Test func anInstalledRevisionIsReturnedAndTheFolderLeftAlone() async throws {
    let root = TestSupport.tempDir()
    let base = try publishedArchive()
    let first = try await store(root).ensureModel(archiveAt: base, revision: "r1")
    let (folder, _) = try Self.unpacked(), compiler = FakeCompiler()
    let again = try await store(root, compiler: compiler).adopt(folder: folder, repo: "local", revision: "r1")
    #expect(again == first && compiler.calls.withLock { $0 } == 0)
    #expect(FileManager.default.fileExists(atPath: folder.path))
  }

  @Test func adoptedRevisionNeedsNoDownloadLater() async throws {
    let root = TestSupport.tempDir(), (folder, _) = try Self.unpacked()
    let adopted = try await store(root).adopt(folder: folder, repo: "local", revision: "r1")
    let fetcher = CountingFetcher()
    let later = try await store(root, fetcher: fetcher).ensureModel(archiveAt: try publishedArchive(), revision: "r1")
    #expect(later == adopted && fetcher.count.withLock { $0 } == 0)
  }

  @Test func symlinksAndMissingFilesAreRefusedBeforeAnythingMoves() async throws {
    let (linked, _) = try Self.unpacked()
    try FileManager.default.createSymbolicLink(atPath: linked.appending(path: "tokenizer/x").path,
                                               withDestinationPath: "/etc/hosts")
    await expectError({ _ = try await store().adopt(folder: linked, repo: "local", revision: "r1") }) {
      if case .downloadFailed = $0 { true } else { false }
    }
    #expect(FileManager.default.fileExists(atPath: linked.path))
    let (partial, _) = try Self.unpacked()
    try FileManager.default.removeItem(at: partial.appending(path: "tokenizer/tokenizer_config.json"))
    await expectError({ _ = try await store().adopt(folder: partial, repo: "local", revision: "r1") }) {
      if case .downloadFailed(let e) = $0 { (e as? ManifestError) == .missing("tokenizer/tokenizer_config.json") }
      else { false }
    }
  }

  @Test func aFailedCompileGivesTheFolderBack() async throws {
    let (folder, _) = try Self.unpacked()
    await expectError({ _ = try await store(compiler: FakeCompiler(.fails)).adopt(folder: folder, repo: "local", revision: "r1") }) {
      if case .modelLoadFailed = $0 { true } else { false }
    }
    #expect(FileManager.default.fileExists(atPath: folder.appending(path: "tokenizer/tokenizer.json").path))
  }
}
