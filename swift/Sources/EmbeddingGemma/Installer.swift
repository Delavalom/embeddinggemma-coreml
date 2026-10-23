import Foundation
import Synchronization

/// One install of `repo`/`revision` into a store (see `ModelStore` for the guarantees).
struct Installer: Sendable {
  let store: ModelStore
  let base: URL
  let repo: String
  let revision: String
  let manifest: Data?
  let expectedManifestSHA256: String?
  let progress: (@Sendable (InstallProgress) -> Void)?

  static let requiredFiles = ["\(ModelStore.packageName)/Manifest.json", "\(ModelStore.tokenizerDir)/tokenizer.json",
                              "\(ModelStore.tokenizerDir)/tokenizer_config.json"]

  func run() async throws -> InstalledModel {
    do {
      try prepareRoot()
      if let done = store.installed(repo: repo, revision: revision) { return finished(done) }
      return try await FileLock.withLock(store.root.appending(path: ModelStore.lockName)) {
        // Another process may have finished it while we waited for the lock.
        if let done = store.installed(repo: repo, revision: revision) { return finished(done) }
        return try await install()
      }
    } catch {
      throw EmbeddingGemmaError.storage(error)
    }
  }

  private func finished(_ m: InstalledModel) -> InstalledModel {
    progress?(InstallProgress(completedBytes: 1, totalBytes: 1))
    return m
  }

  private func prepareRoot() throws {
    var root = store.root
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    var values = URLResourceValues()
    values.isExcludedFromBackup = true  // re-downloadable; keep it out of iCloud and device backups
    try root.setResourceValues(values)
  }

  private func install() async throws -> InstalledModel {
    let fm = FileManager.default
    let repoDir = store.repoFolder(repo)
    let partial = repoDir.appending(path: "\(revision).partial-\(UUID().uuidString)")
    let files = partial.appending(path: "files")
    try fm.createDirectory(at: files, withIntermediateDirectories: true)
    var marker = InstallMarker(repo: repo, revision: revision, filesSHA256: nil, complete: false)
    try marker.write(partial)

    let manifestBytes = try await manifestData(into: partial)
    let sha = Checksums.sha256(data: manifestBytes)
    if let expected = expectedManifestSHA256, expected != sha {
      throw EmbeddingGemmaError.checksumMismatch(file: FilesManifest.fileName)
    }
    let entries: [ManifestEntry]
    do {
      entries = try FilesManifest.decode(manifestBytes)
      for required in Self.requiredFiles where !entries.contains(where: { $0.path == required }) {
        throw ManifestError.missing(required)
      }
    } catch { throw EmbeddingGemmaError.downloadFailed(underlying: error) }
    marker.filesSHA256 = sha
    try marker.write(partial)

    try adoptStalePartials(into: files, repoDir: repoDir, except: partial)
    let total = entries.reduce(Int64(0)) { $0 + $1.size }
    if let available = store.availableCapacity(store.root), available < ModelStore.spaceFactor * total {
      throw EmbeddingGemmaError.insufficientStorage(required: ModelStore.spaceFactor * total, available: available)
    }

    var done: Int64 = 0
    for entry in entries {
      try Task.checkCancellation()
      let before = done
      try await VerifiedDownload.fetch(entry, base: base, into: files, fetcher: store.fetcher) { bytes in
        progress?(InstallProgress(completedBytes: before + min(bytes, entry.size), totalBytes: total))
      }
      done += entry.size
      progress?(InstallProgress(completedBytes: done, totalBytes: total))
    }
    try removeUnlisted(in: files, keeping: Set(entries.map(\.path)))

    let compiled = partial.appending(path: ModelStore.compiledName)
    let tmp: URL
    do { tmp = try await store.compiler(files.appending(path: ModelStore.packageName)) } catch {
      if error is CancellationError { throw error }
      throw EmbeddingGemmaError.isOutOfSpace(error) ? EmbeddingGemmaError.storageFailed(underlying: error)
        : EmbeddingGemmaError.modelLoadFailed(underlying: error)
    }
    try fm.moveItem(at: tmp, to: compiled)
    try fm.moveItem(at: files.appending(path: ModelStore.tokenizerDir), to: partial.appending(path: ModelStore.tokenizerDir))
    try fm.removeItem(at: files)  // the package is compiled; drop the download
    marker.complete = true
    try marker.write(partial)
    return try promote(partial)
  }

  /// `files.json`: the supplied bytes, or fetched fresh into the attempt folder (never reused: it is what every
  /// other file is checked against).
  private func manifestData(into partial: URL) async throws -> Data {
    if let manifest { return manifest }
    let url = partial.appending(path: FilesManifest.fileName)
    do {
      try await store.fetcher.fetch(base.appending(path: FilesManifest.fileName), to: url,
                                    resumeFile: partial.appending(path: "files.json.resume"), progress: nil)
      return try Data(contentsOf: url)
    } catch is CancellationError { throw CancellationError() } catch {
      throw EmbeddingGemmaError.downloadFailed(underlying: error)
    }
  }

  /// Earlier attempts of this revision that died (we hold the per-revision task and the root lock, so none is
  /// running): reuse their downloaded files, then delete them. Only folders with our marker are touched.
  private func adoptStalePartials(into files: URL, repoDir: URL, except current: URL) throws {
    let fm = FileManager.default
    let prefix = "\(revision).partial-"
    for old in (try? fm.contentsOfDirectory(at: repoDir, includingPropertiesForKeys: nil)) ?? []
    where old.lastPathComponent.hasPrefix(prefix) && old.lastPathComponent != current.lastPathComponent {
      guard let m = InstallMarker.read(old), m.repo == repo, m.revision == revision else { continue }
      let oldFiles = old.appending(path: "files")
      if fm.fileExists(atPath: oldFiles.path), (try? fm.contentsOfDirectory(atPath: files.path))?.isEmpty ?? true {
        try fm.removeItem(at: files)
        try fm.moveItem(at: oldFiles, to: files)
      }
      try fm.removeItem(at: old)
    }
  }

  /// Deletes files under `dir` that `files.json` does not list (leftovers of another manifest, temp files), so the
  /// compiled package is exactly the published one.
  private func removeUnlisted(in dir: URL, keeping listed: Set<String>) throws {
    let fm = FileManager.default
    let prefix = dir.standardizedFileURL.resolvingSymlinksInPath().path + "/"
    guard let walker = fm.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey]) else { return }
    var extra: [URL] = []
    for case let url as URL in walker where (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true {
      let rel = String(url.resolvingSymlinksInPath().path.dropFirst(prefix.count))
      if !listed.contains(rel) { extra.append(url) }
    }
    for url in extra { try fm.removeItem(at: url) }
  }

  /// Renames the finished attempt to `<revision>/`. A valid install that appeared meanwhile wins and is never
  /// removed; an incomplete folder of ours is replaced; a foreign folder in the way is an error.
  private func promote(_ partial: URL) throws -> InstalledModel {
    let fm = FileManager.default
    if let existing = store.installed(repo: repo, revision: revision) {
      try fm.removeItem(at: partial)
      return existing
    }
    let final = store.revisionFolder(repo: repo, revision: revision)
    if fm.fileExists(atPath: final.path) {
      guard InstallMarker.read(final) != nil else {
        throw EmbeddingGemmaError.storageFailed(underlying: CocoaError(.fileWriteFileExists, userInfo: [NSFilePathErrorKey: final.path]))
      }
      try fm.removeItem(at: final)
    }
    try fm.moveItem(at: partial, to: final)
    guard let m = store.installed(repo: repo, revision: revision) else {
      throw EmbeddingGemmaError.storageFailed(underlying: CocoaError(.fileReadCorruptFile))
    }
    return m
  }
}
