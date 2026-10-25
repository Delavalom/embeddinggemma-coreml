import CoreML
import Foundation

/// Download progress of an install, in bytes of verified files. Delivered on an arbitrary thread.
public struct InstallProgress: Sendable, Equatable {
  public let completedBytes: Int64
  public let totalBytes: Int64
  public var fraction: Double { totalBytes > 0 ? min(1, Double(completedBytes) / Double(totalBytes)) : 1 }
}

/// A model revision that is downloaded, verified and compiled.
public struct InstalledModel: Sendable, Equatable {
  public let repo: String
  public let revision: String
  /// `<root>/<repo with "/" as "--">/<revision>`.
  public let folder: URL
  /// The compiled multifunction model (`model.mlmodelc`).
  public var compiledModel: URL { folder.appending(path: ModelStore.compiledName) }
  /// The folder with `tokenizer.json` and `tokenizer_config.json`.
  public var tokenizer: URL { folder.appending(path: ModelStore.tokenizerDir) }
}

/// Downloads, verifies and compiles the published model into `<root>/<repo>/<revision>/`.
///
/// From the Hugging Face Hub (a pinned commit) it downloads `manifest.json` (`ArchiveManifest`) and the one archive
/// it names: an LZFSE Apple Archive of `EmbeddingGemma2.mlpackage/**` and `tokenizer/{tokenizer.json,
/// tokenizer_config.json}`, resumable, checked against its size and SHA-256, then validated (no symlinks, no absolute
/// or ".." paths) and extracted with the AppleArchive framework. A local folder may instead hold the plain layout,
/// `files.json` plus those files, each checked on its own. Then the package is compiled with
/// `MLModel.compileModel(at:)`.
///
/// - Every attempt works in its own `<revision>.partial-<uuid>/` and renames it to `<revision>/` only after the
///   compile succeeds. Files that already verified in an earlier failed attempt of the same revision are reused.
/// - Concurrent calls for the same repo and revision share one install; installs and cleanups are serialized across
///   processes with a lock on `<root>/.lock`.
/// - `ensureModel` never deletes another revision. Old revisions are removed only by `removeRevisions(except:)`,
///   which deletes folders this store created (they carry a `.egcm-install.json` marker) and never one that a live
///   `EmbeddingGemma` in this process has loaded.
public struct ModelStore: Sendable {
  public static let packageName = "EmbeddingGemma2.mlpackage"
  public static let tokenizerDir = "tokenizer"
  static let compiledName = "model.mlmodelc"
  static let lockName = ".lock"
  /// Free space required before downloading, as a multiple of the download (the files, the compiled model, slack).
  static let spaceFactor: Int64 = 3

  public let root: URL
  let fetcher: any FileFetcher
  let compiler: @Sendable (URL) async throws -> URL
  let availableCapacity: @Sendable (URL) -> Int64?

  /// macOS: `Application Support/<bundle identifier or "egcm">/EmbeddingGemma`, so apps, tests and command-line
  /// tools that are not sandboxed do not share a folder. iOS: the app's own `Application Support/EmbeddingGemma`.
  public static var defaultRoot: URL {
    #if os(macOS)
      URL.applicationSupportDirectory.appending(path: Bundle.main.bundleIdentifier ?? "egcm")
        .appending(path: "EmbeddingGemma")
    #else
      URL.applicationSupportDirectory.appending(path: "EmbeddingGemma")
    #endif
  }

  public init(root: URL = ModelStore.defaultRoot) {
    self.init(root: root, fetcher: URLSessionFetcher())
  }

  package init(root: URL, fetcher: any FileFetcher,
               compiler: @escaping @Sendable (URL) async throws -> URL = { try await MLModel.compileModel(at: $0) },
               availableCapacity: @escaping @Sendable (URL) -> Int64? = ModelStore.volumeCapacity) {
    self.root = root
    self.fetcher = fetcher
    self.compiler = compiler
    self.availableCapacity = availableCapacity
  }

  public static func huggingFaceBase(repo: String, revision: String) -> URL {
    URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/")!
  }

  /// `repo` must be `owner/name` and `revision` a full 40-character commit SHA (never a branch: the checksums in
  /// `files.json` are only meaningful for a fixed commit).
  static func validate(repo: String, revision: String) throws {
    guard repo.wholeMatch(of: /[A-Za-z0-9_.\-]+\/[A-Za-z0-9_.\-]+/) != nil else {
      throw EmbeddingGemmaError.invalidSource("repo must be owner/name, got \"\(repo)\"")
    }
    guard revision.wholeMatch(of: /[0-9a-f]{40}/) != nil else {
      throw EmbeddingGemmaError.invalidSource("revision must be a 40-character lowercase commit SHA, got \"\(revision)\"")
    }
  }

  /// The model of `repo` at `revision` from the Hugging Face Hub: `manifest.json` and the one archive it names.
  /// `manifestSHA256`, when given, pins `manifest.json`.
  public func ensureModel(repo: String, revision: String, manifestSHA256: String? = nil,
                          progress: (@Sendable (InstallProgress) -> Void)? = nil) async throws -> InstalledModel {
    try Self.validate(repo: repo, revision: revision)
    return try await ensureModel(archiveAt: Self.huggingFaceBase(repo: repo, revision: revision), repo: repo,
                                 revision: revision, manifestSHA256: manifestSHA256, progress: progress)
  }

  /// The archive published at `baseURL` (https:// or file://: `manifest.json` + the archive), as `repo`/`revision`.
  package func ensureModel(archiveAt baseURL: URL, repo: String = "local", revision: String,
                           manifestSHA256: String? = nil,
                           progress: (@Sendable (InstallProgress) -> Void)? = nil) async throws -> InstalledModel {
    try await install(baseURL: baseURL, repo: repo, revision: revision, layout: .archive,
                      expectedSHA256: manifestSHA256, progress: progress)
  }

  /// Installs `folder`, an already unpacked and verified layout (`EmbeddingGemma2.mlpackage/**` and
  /// `tokenizer/{tokenizer.json,tokenizer_config.json}`, for example the app archive extracted by the caller), as
  /// `repo`/`revision` without downloading or copying anything.
  ///
  /// - The folder is **moved** into the store (a rename: it must be on the same volume as `root`, else
  ///   `.storageFailed`), and its package is compiled there. Free space needed: the compiled model only, about the
  ///   package's size (checked first, `.insufficientStorage`); the package is deleted once compiled, so afterwards
  ///   the install takes about what the folder took.
  /// - If `repo`/`revision` is already installed, that install is returned and `folder` is left untouched.
  /// - Symlinks and a missing package or tokenizer file are refused before anything moves.
  public func adopt(folder: URL, repo: String, revision: String) async throws -> InstalledModel {
    try await install(baseURL: folder, repo: repo, revision: revision, layout: .adopt(folder), expectedSHA256: nil,
                      progress: nil)
  }

  /// The plain files published at `baseURL` (https:// or file://), installed as `repo`/`revision`. `manifest`
  /// supplies the bytes of `files.json` instead of fetching them (a local folder hashes and parses the same bytes).
  package func ensureModel(baseURL: URL, repo: String = "local", revision: String, manifest: Data? = nil,
                           filesSHA256: String? = nil,
                           progress: (@Sendable (InstallProgress) -> Void)? = nil) async throws -> InstalledModel {
    try await install(baseURL: baseURL, repo: repo, revision: revision, layout: .files(manifest: manifest),
                      expectedSHA256: filesSHA256, progress: progress)
  }

  private func install(baseURL: URL, repo: String, revision: String, layout: Installer.Layout,
                       expectedSHA256: String?,
                       progress: (@Sendable (InstallProgress) -> Void)?) async throws -> InstalledModel {
    guard FilesManifest.isSafe(repo), FilesManifest.isSafe(revision), !revision.contains("/"),
          !revision.hasPrefix("."), !revision.contains(".partial") else {
      throw EmbeddingGemmaError.invalidSource("unsafe repo or revision: \(repo) \(revision)")
    }
    let folder = revisionFolder(repo: repo, revision: revision)
    let key = StoreRegistry.key(folder)
    let job = Installer(store: self, base: baseURL, repo: repo, revision: revision, layout: layout,
                        expectedManifestSHA256: expectedSHA256, progress: progress)
    let (task, created) = StoreRegistry.inFlight.withLock { tasks -> (Task<InstalledModel, any Error>, Bool) in
      if let t = tasks[key] { return (t, false) }
      let t = Task { try await job.run() }
      tasks[key] = t
      return (t, true)
    }
    defer { if created { _ = StoreRegistry.inFlight.withLock { $0.removeValue(forKey: key) } } }
    // The caller that started the install also cancels it; later callers only stop waiting.
    return try await withTaskCancellationHandler { try await task.value } onCancel: { if created { task.cancel() } }
  }

  func repoFolder(_ repo: String) -> URL { root.appending(path: repo.replacingOccurrences(of: "/", with: "--")) }

  func revisionFolder(repo: String, revision: String) -> URL { repoFolder(repo).appending(path: revision) }

  /// The installed model for `repo`/`revision`, if its folder is complete.
  public func installed(repo: String, revision: String) -> InstalledModel? {
    let folder = revisionFolder(repo: repo, revision: revision)
    guard let marker = InstallMarker.read(folder), marker.complete, marker.repo == repo, marker.revision == revision
    else { return nil }
    let m = InstalledModel(repo: repo, revision: revision, folder: folder)
    let fm = FileManager.default
    guard fm.fileExists(atPath: m.compiledModel.path),
          fm.fileExists(atPath: m.tokenizer.appending(path: "tokenizer.json").path) else { return nil }
    return m
  }

  /// Deletes every folder this store created (complete revisions and leftover partial attempts) except `keep`
  /// and revisions a live `EmbeddingGemma` in this process has loaded. Foreign files are never touched.
  /// Returns the deleted folders.
  @discardableResult
  public func removeRevisions(except keep: [InstalledModel] = []) async throws -> [URL] {
    guard FileManager.default.fileExists(atPath: root.path) else { return [] }
    let kept = Set(keep.map { StoreRegistry.key($0.folder) })
    return try await FileLock.withLock(root.appending(path: Self.lockName)) {
      var removed: [URL] = []
      let fm = FileManager.default
      for repoDir in (try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? [] {
        for folder in (try? fm.contentsOfDirectory(at: repoDir, includingPropertiesForKeys: nil)) ?? [] {
          let k = StoreRegistry.key(folder)
          guard InstallMarker.read(folder) != nil, !kept.contains(k), !StoreRegistry.isInUse(folder),
                StoreRegistry.inFlight.withLock({ $0[k] == nil }) else { continue }
          if folder.lastPathComponent.contains(".partial-"),
             StoreRegistry.inFlight.withLock({ $0[StoreRegistry.key(repoDir.appending(path: Self.revisionOf(partial: folder)))] != nil }) {
            continue  // an install of that revision is running in this process
          }
          do { try fm.removeItem(at: folder) } catch { throw EmbeddingGemmaError.storage(error) }
          removed.append(folder)
        }
      }
      return removed
    }
  }

  static func revisionOf(partial: URL) -> String {
    String(partial.lastPathComponent.split(separator: ".partial-", maxSplits: 1).first ?? "")
  }

  /// Free space for "important" data (what the system would free up for us), else plain free space.
  package static let volumeCapacity: @Sendable (URL) -> Int64? = { url in
    let v = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey])
    return v?.volumeAvailableCapacityForImportantUsage ?? v?.volumeAvailableCapacity.map(Int64.init)
  }
}
