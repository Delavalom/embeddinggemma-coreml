import Foundation

extension Installer {
  /// Moves `folder` (an unpacked, verified layout: `EmbeddingGemma2.mlpackage/**` and `tokenizer/`) to `files`.
  /// A rename, so it must be on the store's volume; nothing is copied. Needs free space for the compiled model only
  /// (about the size of the package, which is deleted once it is compiled).
  func adopt(_ folder: URL, into files: URL) throws {
    let fm = FileManager.default
    for required in Self.requiredFiles where !fm.fileExists(atPath: folder.appending(path: required).path) {
      throw EmbeddingGemmaError.downloadFailed(underlying: ManifestError.missing(required))
    }
    let package = folder.appending(path: ModelStore.packageName)
    var packageBytes: Int64 = 0
    let keys: Set<URLResourceKey> = [.isSymbolicLinkKey, .isRegularFileKey, .fileSizeKey]
    if let walker = fm.enumerator(at: folder, includingPropertiesForKeys: Array(keys)) {
      for case let url as URL in walker {
        let v = try url.resourceValues(forKeys: keys)
        if v.isSymbolicLink == true {
          throw EmbeddingGemmaError.downloadFailed(underlying: ModelArchiveError.unsafeEntry(url.lastPathComponent))
        }
        if v.isRegularFile == true, url.standardizedFileURL.path.hasPrefix(package.standardizedFileURL.path + "/") {
          packageBytes += Int64(v.fileSize ?? 0)
        }
      }
    }
    if let available = store.availableCapacity(store.root), available < packageBytes {
      throw EmbeddingGemmaError.insufficientStorage(required: packageBytes, available: available)
    }
    do {
      try fm.removeItem(at: files)
      try fm.moveItem(at: folder, to: files)
    } catch {
      throw EmbeddingGemmaError.storageFailed(underlying: error)  // e.g. another volume: adopt never copies
    }
  }
}
