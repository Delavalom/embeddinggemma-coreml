import AppleArchive
import Foundation
import System

/// `manifest.json` at the root of the published Hugging Face repo: the one archive an app downloads.
///
/// ```json
/// {"format": 1, "archive": "EmbeddingGemma2.aar", "size": 367001600, "sha256": "<64 lowercase hex>",
///  "unpacked_size": 485000000, "revision": "v1.0.0", "image_budget": 70, "functions": ["embed_image_70_wide", ...]}
/// ```
/// `unpacked_size`, `revision`, `image_budget` and `functions` are optional; unknown keys are ignored. Written by
/// `egcm.published.write_app_archive` (s7). Since ADR-0010 each image budget is published as its own package under
/// `b<budget>/`, whose manifest declares `image_budget` (70 or 280); a manifest without it is a 14-function package.
package struct ArchiveManifest: Codable, Sendable, Equatable {
  package static let fileName = "manifest.json"
  package static let format = 1

  package let format: Int
  /// A single path component next to `manifest.json`.
  package let archive: String
  package let size: Int64
  package let sha256: String
  package let unpackedSize: Int64?
  package let revision: String?
  package let functions: [String]?
  /// The one image budget this package serves (70 or 280), or nil for a package with every budget.
  package let imageBudget: Int?

  enum CodingKeys: String, CodingKey {
    case format, archive, size, sha256, revision, functions
    case unpackedSize = "unpacked_size"
    case imageBudget = "image_budget"
  }

  /// The image budgets a package can be split by.
  package static let budgets = [70, 280]

  /// The functions every 14-function model must have: the four text lengths and the ten image buckets.
  package static var requiredFunctions: [String] { requiredFunctions(budget: nil) }

  /// The functions a package for `budget` must have: the four text lengths and that budget's five image buckets
  /// (all ten when `budget` is nil).
  package static func requiredFunctions(budget: Int?) -> [String] {
    EmbeddingGemma.textFunctions + Bucket.all.filter { budget == nil || $0.budget == budget }.map(\.functionName)
  }

  /// The archive as a download entry (`VerifiedDownload` checks its size and SHA-256).
  var entry: ManifestEntry { ManifestEntry(path: archive, size: size, sha256: sha256) }

  /// Parses and checks `manifest.json`: known format, a safe single-component archive name, a lowercase SHA-256,
  /// `image_budget` (when present) 70 or 280, and, when `functions` is present, every function that budget needs
  /// (all 14 without `image_budget`).
  package static func decode(_ data: Data) throws -> ArchiveManifest {
    let m = try JSONDecoder().decode(ArchiveManifest.self, from: data)
    guard m.format == format else { throw ModelArchiveError.unsupportedFormat(m.format) }
    guard FilesManifest.isSafe(m.archive), !m.archive.contains("/"), !FilesManifest.isReserved(m.archive),
          m.archive != fileName else { throw ManifestError.unsafePath(m.archive) }
    guard m.sha256.count == 64, m.sha256.allSatisfy({ $0.isHexDigit && !$0.isUppercase }), m.size > 0,
          (m.unpackedSize ?? 0) >= 0 else { throw ManifestError.badHash(m.archive) }
    if let b = m.imageBudget, !budgets.contains(b) { throw ManifestError.unsupportedImageBudget(b) }
    if let fns = m.functions,
       let missing = requiredFunctions(budget: m.imageBudget).first(where: { !fns.contains($0) }) {
      throw ManifestError.missing(missing)
    }
    return m
  }

  /// Checks that this package serves `budget`: a declared `image_budget` must equal it, and with `mustDeclare` (a
  /// per-budget folder `b<budget>/`) it must be declared at all.
  package func check(serves budget: Int, mustDeclare: Bool) throws {
    if imageBudget == nil && !mustDeclare { return }
    guard imageBudget == budget else { throw ManifestError.imageBudgetMismatch(expected: budget, found: imageBudget) }
  }
}

/// The image budget a `manifest.json` must serve (`ArchiveManifest.check(serves:mustDeclare:)`).
package struct BudgetRequirement: Sendable, Equatable {
  package let budget: Int
  /// True for a per-budget folder (`b<budget>/`): its manifest must declare `image_budget`.
  package let mustDeclare: Bool

  package init(budget: Int, mustDeclare: Bool) {
    self.budget = budget
    self.mustDeclare = mustDeclare
  }
}

package enum ModelArchiveError: Error, Hashable {
  case unsupportedFormat(Int)
  /// An entry that could write outside the destination, or is not a regular file or directory (symlinks too).
  case unsafeEntry(String)
  /// Not a readable LZFSE Apple Archive (AppleArchive's decoder also refuses absolute and ".." paths this way).
  case unreadable(String)
}

/// The app archive: Apple Archive, LZFSE, with `EmbeddingGemma2.mlpackage/**` and `tokenizer/` at its root.
package enum ModelArchive {
  /// Every entry's path, after checking that each is a relative path without "." or ".." components and is a
  /// regular file or a directory. Reads the archive without writing anything.
  package static func validate(_ archive: URL) throws -> [String] {
    var paths: [String] = []
    do {
      try withDecoder(archive) { decoder in
        while let header = try decoder.readHeader() {
          paths.append(try check(path: header.entryPath, type: header.entryType))
        }
      }
    } catch let e as ModelArchiveError { throw e } catch {
      throw ModelArchiveError.unreadable(archive.lastPathComponent)  // the decoder itself refused it
    }
    return paths
  }

  /// Validates `archive`, then extracts it into `directory` (created; must not exist yet).
  package static func extract(_ archive: URL, into directory: URL) throws {
    _ = try validate(archive)
    let fm = FileManager.default
    guard !fm.fileExists(atPath: directory.path) else {
      throw CocoaError(.fileWriteFileExists, userInfo: [NSFilePathErrorKey: directory.path])
    }
    try fm.createDirectory(at: directory, withIntermediateDirectories: true)
    let rejected = RejectedEntry()
    // The filter repeats the checks while extracting, so the bytes written are the bytes validated.
    let filter: ArchiveHeader.EntryFilter = { _, path, data in
      guard case .header(let h)? = data else { return .ok }
      if (try? check(path: h.entryPath ?? path, type: h.entryType)) == nil {
        rejected.path = path.string
        return .cancel
      }
      return .ok
    }
    do {
      try withDecoder(archive) { decoder in
        try ArchiveStream.withExtractStream(extractingTo: FilePath(directory.path),
                                            flags: [.ignoreOperationNotPermitted]) { extractor in
          _ = try ArchiveStream.process(readingFrom: decoder, writingTo: extractor, selectUsing: filter)
        }
      }
    } catch {
      if let p = rejected.path { throw ModelArchiveError.unsafeEntry(p) }
      throw error
    }
  }

  /// Writes the contents of `directory` (not its name) as an LZFSE archive: tests and tooling.
  package static func create(from directory: URL, to archive: URL) throws {
    guard let keys = ArchiveHeader.FieldKeySet("TYP,PAT,DAT,MOD,SIZ") else { throw ModelArchiveError.unreadable("keys") }
    try ArchiveByteStream.withFileStream(path: FilePath(archive.path), mode: .writeOnly,
                                         options: [.create, .truncate], permissions: [.ownerReadWrite]) { file in
      try ArchiveByteStream.withCompressionStream(using: .lzfse, writingTo: file) { compressed in
        try ArchiveStream.withEncodeStream(writingTo: compressed) { encoder in
          try encoder.writeDirectoryContents(archiveFrom: FilePath(directory.path), keySet: keys)
        }
      }
    }
  }

  static func withDecoder(_ archive: URL, _ body: (ArchiveStream) throws -> Void) throws {
    guard FileManager.default.isReadableFile(atPath: archive.path) else {
      throw ModelArchiveError.unreadable(archive.lastPathComponent)
    }
    try ArchiveByteStream.withFileStream(path: FilePath(archive.path), mode: .readOnly, options: [],
                                         permissions: []) { file in
      try ArchiveByteStream.withDecompressionStream(readingFrom: file) { raw in
        try ArchiveStream.withDecodeStream(readingFrom: raw) { try body($0) }
      }
    }
  }

  /// The root entry has an empty path; everything else must be a plain relative path.
  static func check(path: FilePath?, type: ArchiveHeader.EntryType?) throws -> String {
    let p = path?.string ?? ""
    guard type == .regularFile || type == .directory else { throw ModelArchiveError.unsafeEntry(p) }
    if p.isEmpty, type == .directory { return p }
    guard FilesManifest.isSafe(p) else { throw ModelArchiveError.unsafeEntry(p) }
    return p
  }
}

/// The path the extraction filter refused (the filter runs synchronously inside `process`).
private final class RejectedEntry: @unchecked Sendable {
  var path: String?
}
