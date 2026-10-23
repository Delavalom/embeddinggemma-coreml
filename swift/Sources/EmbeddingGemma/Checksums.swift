import CryptoKit
import Foundation

package enum Checksums {
  /// Lowercase hex SHA-256 of a file, streamed in 1 MB chunks.
  package static func sha256(file: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: file)
    defer { try? handle.close() }
    var hasher = SHA256()
    while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
    return hex(hasher.finalize())
  }

  package static func sha256(data: Data) -> String { hex(SHA256.hash(data: data)) }

  private static func hex(_ digest: SHA256.Digest) -> String { digest.map { String(format: "%02x", $0) }.joined() }
}

/// One line of `files.json`: a file of the published model folder, relative to its root.
package struct ManifestEntry: Codable, Sendable, Hashable {
  package let path: String
  package let size: Int64
  package let sha256: String

  package init(path: String, size: Int64, sha256: String) {
    self.path = path
    self.size = size
    self.sha256 = sha256
  }
}

package enum ManifestError: Error, Equatable {
  case unsafePath(String)
  case reservedName(String)
  case duplicatePath(String)
  case badHash(String)
  case missing(String)
}

/// `files.json`: the list of files in a published model folder (the uncompressed `.mlpackage` and the tokenizer),
/// each with its size and SHA-256. There is no system unzip API on iOS, so the package is published as plain files
/// and every file is downloaded and verified on its own. Publisher tooling (`make`, `write`) is package-internal.
package enum FilesManifest {
  package static let fileName = "files.json"
  /// Suffixes of the store's own temporary files next to a download.
  static let reservedSuffixes = [".download", ".resume"]

  /// Every regular file under `root` except `files.json` and dotfiles, sorted by path.
  package static func make(root: URL) throws -> [ManifestEntry] {
    let root = root.standardizedFileURL.resolvingSymlinksInPath()
    let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey]
    guard let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: keys,
                                                      options: [.skipsHiddenFiles]) else { return [] }
    var out: [ManifestEntry] = []
    let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
    for case let url as URL in walker {
      let values = try url.resourceValues(forKeys: Set(keys))
      guard values.isRegularFile == true else { continue }
      let path = String(url.resolvingSymlinksInPath().path.dropFirst(prefix.count))
      if path == fileName { continue }
      out.append(ManifestEntry(path: path, size: Int64(values.fileSize ?? 0), sha256: try Checksums.sha256(file: url)))
    }
    return out.sorted { $0.path < $1.path }
  }

  /// Writes `root/files.json` for the folder as it is now.
  @discardableResult
  package static func write(root: URL) throws -> [ManifestEntry] {
    let entries = try make(root: root)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(entries).write(to: root.appending(path: fileName), options: .atomic)
    return entries
  }

  /// Parses `files.json` and rejects anything that could write outside the install folder, collide with the
  /// store's own files, appear twice, or is not a lowercase SHA-256.
  package static func decode(_ data: Data) throws -> [ManifestEntry] {
    let entries = try JSONDecoder().decode([ManifestEntry].self, from: data)
    var seen = Set<String>()
    for e in entries {
      guard isSafe(e.path) else { throw ManifestError.unsafePath(e.path) }
      guard !isReserved(e.path) else { throw ManifestError.reservedName(e.path) }
      guard seen.insert(e.path).inserted else { throw ManifestError.duplicatePath(e.path) }
      guard e.sha256.count == 64, e.sha256.allSatisfy({ $0.isHexDigit && !$0.isUppercase }), e.size >= 0 else {
        throw ManifestError.badHash(e.path)
      }
    }
    return entries
  }

  /// A relative path of plain components: no leading "/", no "", ".", or ".." component.
  static func isSafe(_ path: String) -> Bool {
    guard !path.isEmpty, !path.hasPrefix("/") else { return false }
    return path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !["", ".", ".."].contains($0) }
  }

  /// `files.json` itself, hidden components (the store's markers and locks are dotfiles), and the store's
  /// temporary-file suffixes.
  static func isReserved(_ path: String) -> Bool {
    if path == fileName { return true }
    return path.split(separator: "/").contains { c in
      c.hasPrefix(".") || reservedSuffixes.contains { c.hasSuffix($0) }
    }
  }
}
