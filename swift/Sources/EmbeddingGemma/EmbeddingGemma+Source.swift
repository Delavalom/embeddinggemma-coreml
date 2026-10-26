import Foundation

extension EmbeddingGemma {
  /// Installs (or finds installed) what `source` names for image `budget` (70 or 280), and returns the install with
  /// the revision `modelRevision` reports (ADR-0010):
  ///
  /// - `.huggingFace`: the `b<budget>/` package, installed as `<revision>-b<budget>`; reports `revision`.
  /// - `.file(folder)`: `folder/b<budget>/` if that folder exists, else `folder` (a 14-function layout); reports and
  ///   installs as `file-<sha256 prefix>` of the `files.json` (or `manifest.json`) of the folder used.
  /// - `.installed`: unchanged; the caller passes the layout of the budget it loads.
  package static func install(_ source: Source, budget: Int, store: ModelStore,
                              progress: (@Sendable (InstallProgress) -> Void)?) async throws
    -> (installed: InstalledModel, revision: String) {
    switch source {
    case .huggingFace(let repo, let revision):
      let pinned = repo == defaultRepo && revision == defaultRevision ? defaultManifestSHA256[budget] : nil
      let m = try await store.ensureModel(repo: repo, revision: revision, budget: budget, manifestSHA256: pinned,
                                          progress: progress)
      return (m, revision)
    case .installed(let folder, let repo, let revision):
      return (try await store.adopt(folder: folder, repo: repo, revision: revision), revision)
    case .file(let root):
      let (folder, perBudget) = fileFolder(root, budget: budget)
      let files = folder.appending(path: FilesManifest.fileName)
      let archived = folder.appending(path: ArchiveManifest.fileName)
      let fm = FileManager.default
      if !fm.fileExists(atPath: files.path), fm.fileExists(atPath: archived.path) {
        let bytes = try read(archived), sha = Checksums.sha256(data: bytes)
        let revision = "file-" + sha.prefix(16)
        let m = try await store.ensureModel(archiveAt: folder, repo: "local", revision: revision, manifestSHA256: sha,
                                            budget: BudgetRequirement(budget: budget, mustDeclare: perBudget),
                                            progress: progress)
        return (m, revision)
      }
      let bytes = try read(files)
      let revision = "file-" + Checksums.sha256(data: bytes).prefix(16)  // hash and parse the same bytes
      let m = try await store.ensureModel(baseURL: folder, repo: "local", revision: revision, manifest: bytes,
                                          progress: progress)
      return (m, revision)
    }
  }

  /// The folder `.file(root)` installs for `budget`: `root/b<budget>/` when that directory exists (`perBudget`),
  /// else `root` itself.
  package static func fileFolder(_ root: URL, budget: Int) -> (folder: URL, perBudget: Bool) {
    let sub = root.appending(path: ModelStore.budgetFolder(budget), directoryHint: .isDirectory)
    var isDir: ObjCBool = false
    if FileManager.default.fileExists(atPath: sub.path, isDirectory: &isDir), isDir.boolValue { return (sub, true) }
    return (root, false)
  }

  private static func read(_ url: URL) throws -> Data {
    do { return try Data(contentsOf: url) } catch { throw EmbeddingGemmaError.downloadFailed(underlying: error) }
  }
}
