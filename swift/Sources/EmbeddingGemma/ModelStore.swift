import CoreML
import Foundation

/// A model revision that is downloaded, verified and compiled.
public struct InstalledModel: Sendable, Equatable {
  public let revision: String
  /// The compiled multifunction model (`model.mlmodelc`).
  public let compiledModel: URL
  /// The folder with `tokenizer.json` and `tokenizer_config.json`.
  public let tokenizer: URL
}

/// Downloads, verifies and compiles the published model into `<root>/<revision>/`.
///
/// The published folder (a Hugging Face repo at a pinned commit, or a local `file://` folder) has the same layout:
/// `files.json`, `EmbeddingGemma2.mlpackage/**` (uncompressed) and `tokenizer/{tokenizer.json,tokenizer_config.json}`.
/// Each file is downloaded on its own (resumable), checked against its size and SHA-256 in `files.json`, then the
/// package is compiled with `MLModel.compileModel(at:)`. Work happens in `<revision>.partial/`; it is renamed to
/// `<revision>/` only after the compile succeeds, and only then are the other revisions deleted. A failed or
/// interrupted install leaves the previous revision in place, and a retry reuses every file that already verifies.
public struct ModelStore: Sendable {
  public static let packageName = "EmbeddingGemma2.mlpackage"
  public static let tokenizerDir = "tokenizer"
  static let compiledName = "model.mlmodelc"

  public let root: URL
  let fetcher: any FileFetcher
  let compiler: @Sendable (URL) async throws -> URL

  /// Application Support/EmbeddingGemma.
  public static var defaultRoot: URL { URL.applicationSupportDirectory.appending(path: "EmbeddingGemma") }

  public init(root: URL = ModelStore.defaultRoot, fetcher: any FileFetcher = URLSessionFetcher(),
              compiler: @escaping @Sendable (URL) async throws -> URL = { try await MLModel.compileModel(at: $0) }) {
    self.root = root
    self.fetcher = fetcher
    self.compiler = compiler
  }

  public static func huggingFaceBase(repo: String, revision: String) -> URL {
    URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/")!
  }

  /// The model of `repo` at `revision` (a commit SHA), from the Hugging Face Hub.
  public func ensureModel(repo: String, revision: String,
                          progress: (@Sendable (Double) -> Void)? = nil) async throws -> InstalledModel {
    try await ensureModel(baseURL: Self.huggingFaceBase(repo: repo, revision: revision), revision: revision,
                          progress: progress)
  }

  /// The model published at `baseURL` (https:// or file://), installed under the name `revision`.
  /// `progress` gets the verified fraction of bytes, ending at 1.
  public func ensureModel(baseURL: URL, revision: String,
                          progress: (@Sendable (Double) -> Void)? = nil) async throws -> InstalledModel {
    guard FilesManifest.isSafe(revision), !revision.contains("/"), !revision.hasSuffix(".partial") else {
      throw EmbeddingGemmaError.downloadFailed(underlying: ManifestError.unsafePath(revision))
    }
    if let done = installed(revision: revision) {
      progress?(1)
      return done
    }
    let fm = FileManager.default
    let staging = root.appending(path: "\(revision).partial"), files = staging.appending(path: "files")
    try fm.createDirectory(at: files, withIntermediateDirectories: true)

    let manifestURL = files.appending(path: FilesManifest.fileName)
    do {
      try? fm.removeItem(at: manifestURL)  // always fetched fresh; it is what the other files are checked against
      try await fetcher.fetch(baseURL.appending(path: FilesManifest.fileName), to: manifestURL,
                              resumeFile: staging.appending(path: "files.json.resume"))
    } catch is CancellationError { throw CancellationError() } catch {
      throw EmbeddingGemmaError.downloadFailed(underlying: error)
    }
    let entries: [ManifestEntry]
    do {
      entries = try FilesManifest.decode(Data(contentsOf: manifestURL))
      for required in ["\(Self.packageName)/Manifest.json", "\(Self.tokenizerDir)/tokenizer.json",
                       "\(Self.tokenizerDir)/tokenizer_config.json"] where !entries.contains(where: { $0.path == required }) {
        throw ManifestError.missing(required)
      }
    } catch { throw EmbeddingGemmaError.downloadFailed(underlying: error) }

    let total = max(1, entries.reduce(Int64(0)) { $0 + $1.size })
    var done: Int64 = 0
    for entry in entries {
      try Task.checkCancellation()
      try await VerifiedDownload.fetch(entry, base: baseURL, into: files, fetcher: fetcher)
      done += entry.size
      progress?(Double(done) / Double(total) * 0.99)
    }

    let compiled = staging.appending(path: Self.compiledName)
    do {
      let tmp = try await compiler(files.appending(path: Self.packageName))
      try? fm.removeItem(at: compiled)
      try fm.moveItem(at: tmp, to: compiled)
    } catch { throw EmbeddingGemmaError.modelLoadFailed(underlying: error) }

    let tokenizer = staging.appending(path: Self.tokenizerDir)
    try? fm.removeItem(at: tokenizer)
    try fm.moveItem(at: files.appending(path: Self.tokenizerDir), to: tokenizer)
    try fm.removeItem(at: files)  // the package is compiled; drop the download

    let final = root.appending(path: revision)
    try? fm.removeItem(at: final)
    try fm.moveItem(at: staging, to: final)
    for other in try fm.contentsOfDirectory(atPath: root.path) where other != revision {
      try? fm.removeItem(at: root.appending(path: other))
    }
    progress?(1)
    return installed(revision: revision)!
  }

  /// The installed model for `revision`, if its folder is complete.
  public func installed(revision: String) -> InstalledModel? {
    let dir = root.appending(path: revision)
    let m = InstalledModel(revision: revision, compiledModel: dir.appending(path: Self.compiledName),
                           tokenizer: dir.appending(path: Self.tokenizerDir))
    let fm = FileManager.default
    guard fm.fileExists(atPath: m.compiledModel.path),
          fm.fileExists(atPath: m.tokenizer.appending(path: "tokenizer.json").path) else { return nil }
    return m
  }
}
