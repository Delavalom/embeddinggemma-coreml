import AppleArchive
import Foundation
import System
import Testing
@testable import EmbeddingGemma

/// The app download: `manifest.json` + one LZFSE Apple Archive, made here with the same framework.
func publishedArchive(revision: String? = "v1.0.0", functions: [String]? = ArchiveManifest.requiredFunctions,
                      extra: [String: Any] = [:]) throws -> URL {
  let layout = try publishedFolder()
  try FileManager.default.removeItem(at: layout.appending(path: FilesManifest.fileName))
  let base = TestSupport.tempDir().appending(path: "hub")
  try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
  let aar = base.appending(path: "EmbeddingGemma2.aar")
  try ModelArchive.create(from: layout, to: aar)
  try writeManifest(base, archive: aar, revision: revision, functions: functions, extra: extra)
  return base
}

func writeManifest(_ base: URL, archive aar: URL, revision: String? = nil,
                   functions: [String]? = ArchiveManifest.requiredFunctions, extra: [String: Any] = [:]) throws {
  let size = try #require(try aar.resourceValues(forKeys: [.fileSizeKey]).fileSize)
  var m: [String: Any] = ["format": 1, "archive": aar.lastPathComponent, "size": size,
                          "sha256": try Checksums.sha256(file: aar), "unpacked_size": 100_200]
  if let revision { m["revision"] = revision }
  if let functions { m["functions"] = functions }
  m.merge(extra) { $1 }
  try JSONSerialization.data(withJSONObject: m, options: [.sortedKeys]).write(to: base.appending(path: "manifest.json"))
}

func manifestJSON(_ fields: [String: Any]) throws -> Data {
  var m: [String: Any] = ["format": 1, "archive": "EmbeddingGemma2.aar", "size": 10,
                          "sha256": String(repeating: "a", count: 64)]
  m.merge(fields) { $1 }
  return try JSONSerialization.data(withJSONObject: m)
}

/// Writes an archive whose one entry is a regular file at `path` (the framework's own tools never write "..").
func craftedArchive(path: String, at aar: URL) throws {
  let data = Array("evil".utf8)
  try ArchiveByteStream.withFileStream(path: FilePath(aar.path), mode: .writeOnly, options: [.create, .truncate],
                                       permissions: [.ownerReadWrite]) { file in
    try ArchiveByteStream.withCompressionStream(using: .lzfse, writingTo: file) { compressed in
      try ArchiveStream.withEncodeStream(writingTo: compressed) { encoder in
        let header = ArchiveHeader()
        header.append(.uint(key: ArchiveHeader.FieldKey("TYP"),
                            value: UInt64(ArchiveHeader.EntryType.regularFile.rawValue)))
        header.append(.string(key: ArchiveHeader.FieldKey("PAT"), value: path))
        header.append(.blob(key: ArchiveHeader.FieldKey("DAT"), size: UInt64(data.count)))
        try encoder.writeHeader(header)
        try data.withUnsafeBytes { try encoder.writeBlob(key: ArchiveHeader.FieldKey("DAT"), from: $0) }
      }
    }
  }
}

@Suite struct ArchiveTests {
  @Test func manifestDecodesTheSharedContract() throws {
    let m = try ArchiveManifest.decode(try manifestJSON(["unpacked_size": 20, "revision": "v1.0.0",
                                                         "functions": ArchiveManifest.requiredFunctions + ["x"],
                                                         "future_key": ["ignored": true]]))
    #expect(m.archive == "EmbeddingGemma2.aar" && m.size == 10 && m.unpackedSize == 20 && m.revision == "v1.0.0")
    let bare = try ArchiveManifest.decode(try manifestJSON([:]))  // the optional keys may be absent
    #expect(bare.unpackedSize == nil && bare.revision == nil && bare.functions == nil)
    #expect(ArchiveManifest.requiredFunctions.count == 14 && ArchiveManifest.requiredFunctions.contains("embed_text_128")
            && ArchiveManifest.requiredFunctions.contains("embed_image_280_wide"))
  }

  @Test(arguments: ["../x.aar", "a/b.aar", "/abs.aar", "manifest.json", ".hidden.aar", "x.aar.download", ""])
  func manifestRejectsArchiveNamesOutsideTheRoot(name: String) throws {
    #expect(throws: (any Error).self) { try ArchiveManifest.decode(try manifestJSON(["archive": name])) }
  }

  @Test func manifestRejectsBadFormatHashAndMissingFunctions() throws {
    #expect(throws: ModelArchiveError.unsupportedFormat(2)) { try ArchiveManifest.decode(try manifestJSON(["format": 2])) }
    #expect(throws: ManifestError.badHash("EmbeddingGemma2.aar")) {
      try ArchiveManifest.decode(try manifestJSON(["sha256": String(repeating: "A", count: 64)]))
    }
    let fewer = ArchiveManifest.requiredFunctions.filter { $0 != "embed_image_70_tall" }
    #expect(throws: ManifestError.missing("embed_image_70_tall")) {
      try ArchiveManifest.decode(try manifestJSON(["functions": fewer]))
    }
  }

  @Test func installsFromTheArchive() async throws {
    let base = try publishedArchive(), fetcher = CountingFetcher(), compiler = FakeCompiler()
    let s = store(fetcher: fetcher, compiler: compiler)
    let m = try await s.ensureModel(archiveAt: base, revision: "a1")
    #expect(fetcher.count.withLock { $0 } == 2)  // manifest.json + the archive
    #expect(compiler.lastListing.withLock { $0 }.sorted() == ["Data/com.apple.CoreML/model.mlmodel",
                                                              "Data/com.apple.CoreML/weights/weight.bin", "Manifest.json"])
    #expect(FileManager.default.fileExists(atPath: m.tokenizer.appending(path: "tokenizer_config.json").path))
    let left = try FileManager.default.contentsOfDirectory(atPath: m.folder.path).sorted()
    #expect(left == [".egcm-install.json", "manifest.json", ModelStore.compiledName, ModelStore.tokenizerDir])
    _ = try await s.ensureModel(archiveAt: base, revision: "a1")
    #expect(fetcher.count.withLock { $0 } == 2)
  }

  @Test func pinnedManifestMustMatch() async throws {
    let base = try publishedArchive()
    await expectError({
      _ = try await store().ensureModel(archiveAt: base, revision: "a1", manifestSHA256: String(repeating: "b", count: 64))
    }) { if case .checksumMismatch(file: "manifest.json") = $0 { true } else { false } }
    let good = try Checksums.sha256(file: base.appending(path: "manifest.json"))
    _ = try await store().ensureModel(archiveAt: base, revision: "a1", manifestSHA256: good)
  }

  @Test(arguments: ["corrupt", "short"])
  func corruptOrShortDownloadIsAChecksumMismatch(kind: String) async throws {
    let base = try publishedArchive()
    let aar = base.appending(path: "EmbeddingGemma2.aar")
    var bytes = try Data(contentsOf: aar)
    if kind == "corrupt" { bytes[bytes.count / 2] ^= 0xFF } else { bytes = bytes.prefix(bytes.count - 100) }
    try bytes.write(to: aar)  // after manifest.json recorded the good size and sha256
    let root = TestSupport.tempDir()
    await expectError({ _ = try await store(root).ensureModel(archiveAt: base, revision: "a1") }) {
      if case .checksumMismatch(file: "EmbeddingGemma2.aar") = $0 { true } else { false }
    }
    #expect(store(root).installed(repo: "local", revision: "a1") == nil)
  }

  @Test func aDownloadedArchiveFromADeadAttemptIsReused() async throws {
    let base = try publishedArchive(), root = TestSupport.tempDir()
    let stale = root.appending(path: "local/a1.partial-dead")
    try FileManager.default.createDirectory(at: stale.appending(path: "files"), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: base.appending(path: "EmbeddingGemma2.aar"),
                                     to: stale.appending(path: "files/EmbeddingGemma2.aar"))
    try InstallMarker(repo: "local", revision: "a1", filesSHA256: nil, complete: false).write(stale)
    let fetcher = CountingFetcher()
    _ = try await store(root, fetcher: fetcher).ensureModel(archiveAt: base, revision: "a1")
    #expect(fetcher.count.withLock { $0 } == 1)  // manifest.json only
    #expect(!FileManager.default.fileExists(atPath: stale.path))
  }

  @Test func notEnoughSpaceForArchiveAndContents() async throws {
    let base = try publishedArchive()
    await expectError({ _ = try await store(capacity: 1000).ensureModel(archiveAt: base, revision: "a1") }) {
      if case .insufficientStorage = $0 { true } else { false }
    }
  }

  @Test(arguments: ["../evil", "/tmp/evil", "a/../../evil", "./a"])
  func unsafePathsAreRejectedBeforeAnythingIsWritten(path: String) throws {
    let dir = TestSupport.tempDir(), aar = dir.appending(path: "bad.aar")
    try craftedArchive(path: path, at: aar)
    // Our check refuses "./a"; AppleArchive's decoder already refuses absolute and ".." paths.
    let refused: Set = [ModelArchiveError.unsafeEntry(path), .unreadable("bad.aar")]
    #expect(refused.contains(try #require(throws: ModelArchiveError.self) { _ = try ModelArchive.validate(aar) }))
    let out = dir.appending(path: "deep/out")
    #expect(refused.contains(try #require(throws: ModelArchiveError.self) { try ModelArchive.extract(aar, into: out) }))
    #expect(!FileManager.default.fileExists(atPath: out.path))
    #expect(!FileManager.default.fileExists(atPath: dir.appending(path: "deep/evil").path))
    #expect(!FileManager.default.fileExists(atPath: dir.appending(path: "evil").path))
  }

  @Test func symlinksAreRejected() async throws {
    let layout = try publishedFolder()
    try FileManager.default.createSymbolicLink(atPath: layout.appending(path: "tokenizer/link").path,
                                               withDestinationPath: "/etc/hosts")
    let base = TestSupport.tempDir(), aar = base.appending(path: "EmbeddingGemma2.aar")
    try ModelArchive.create(from: layout, to: aar)
    #expect(throws: ModelArchiveError.unsafeEntry("tokenizer/link")) { _ = try ModelArchive.validate(aar) }
    try writeManifest(base, archive: aar)
    await expectError({ _ = try await store().ensureModel(archiveAt: base, revision: "a1") }) {
      if case .downloadFailed(let e) = $0 { (e as? ModelArchiveError) == .unsafeEntry("tokenizer/link") } else { false }
    }
  }

  @Test func anArchiveWithoutTheTokenizerIsRefused() async throws {
    let layout = try publishedFolder()
    try FileManager.default.removeItem(at: layout.appending(path: "tokenizer"))
    let base = TestSupport.tempDir(), aar = base.appending(path: "EmbeddingGemma2.aar")
    try ModelArchive.create(from: layout, to: aar)
    try writeManifest(base, archive: aar)
    await expectError({ _ = try await store().ensureModel(archiveAt: base, revision: "a1") }) {
      if case .downloadFailed(let e) = $0 { (e as? ManifestError) == .missing("tokenizer/tokenizer.json") } else { false }
    }
  }

  @Test func extractRoundTripsEveryFile() throws {
    let layout = try publishedFolder()
    let aar = TestSupport.tempDir().appending(path: "x.aar"), out = TestSupport.tempDir().appending(path: "x")
    try ModelArchive.create(from: layout, to: aar)
    let paths = try ModelArchive.validate(aar)
    #expect(paths.contains("\(ModelStore.packageName)/Manifest.json"))
    try ModelArchive.extract(aar, into: out)
    let want = try FilesManifest.make(root: layout), got = try FilesManifest.make(root: out)
    #expect(got == want)
  }
}
