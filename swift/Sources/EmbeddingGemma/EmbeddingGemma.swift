import CoreGraphics
import CoreML
import Foundation
import ImageIO

/// On-device EmbeddingGemma 2: text and image embeddings in one vector space, from one multifunction Core ML model.
///
/// ```swift
/// let model = try await EmbeddingGemma.load(.huggingFace(revision: EmbeddingGemma.defaultRevision))
/// let q = try await model.embed(query: "receipts")
/// let v = try await model.embed(image: cgImage)
/// ```
///
/// Embedding calls run concurrently: preprocessing and prediction happen off the actor, which only guards the
/// table of loaded functions. Texts are truncated at 128 tokens (`<bos>`, 126 tokens of prompt + text, `<eos>`).
public actor EmbeddingGemma {
  public enum Source: Sendable {
    /// The published model at `revision` (a 40-character commit SHA) of `repo` (`owner/name`) on the Hugging Face Hub.
    case huggingFace(repo: String = EmbeddingGemma.defaultRepo, revision: String)
    /// A local folder with the plain layout (`files.json`, `EmbeddingGemma2.mlpackage/`, `tokenizer/`), or with
    /// `manifest.json` and the archive it names. Copied into the store (needs about 3x the files in free space).
    case file(URL)
    /// An unpacked, already verified layout (`EmbeddingGemma2.mlpackage/`, `tokenizer/`), e.g. the app archive the
    /// caller downloaded, checked and extracted. Moved into the store as `repo`/`revision`, not copied: it must be
    /// on the store's volume and needs free space for the compiled model only (`ModelStore.adopt`).
    case installed(URL, repo: String = EmbeddingGemma.defaultRepo, revision: String)
  }

  public enum ImageBudget: Sendable {
    /// 280 on A17 Pro and newer iPhones and on Macs, 70 otherwise.
    case auto, tokens70, tokens280

    public var resolved: Int {
      switch self {
      case .auto: DeviceTier.defaultBudget()
      case .tokens70: 70
      case .tokens280: 280
      }
    }
  }

  public static let maxDimensions = 768
  /// The text functions, and the image functions per budget.
  public static let textFunctions = TextEncoder.lengths.map { "embed_text_\($0)" }
  public static func imageFunctions(budget: Int) -> [String] {
    Bucket.all.filter { $0.budget == budget }.map(\.functionName)
  }

  /// The Hugging Face commit SHA of the model (`file-<sha256 prefix of files.json>` for `.file`). Index files store
  /// it so vectors from different models are never mixed.
  public nonisolated let modelRevision: String
  /// Soft tokens per image: 70 or 280.
  public nonisolated let imageBudget: Int
  /// Length of every returned vector.
  public nonisolated let dimensions: Int

  private nonisolated let compiledModel: URL
  private nonisolated let computeUnits: MLComputeUnits
  private nonisolated let encoder: TextEncoder
  private nonisolated let preprocessor: ImagePreprocessor
  /// Keeps the revision folder from being deleted by `ModelStore.removeRevisions` while this instance lives.
  private nonisolated let lease: RevisionLease
  /// One loaded function per name, loaded on first use or by `prewarm`.
  private var functions: [String: Task<LoadedFunction, any Error>] = [:]

  private init(installed: InstalledModel, encoder: TextEncoder, budget: Int, dimensions: Int,
               computeUnits: MLComputeUnits) throws {
    modelRevision = installed.revision
    compiledModel = installed.compiledModel
    lease = RevisionLease(installed.folder)
    self.encoder = encoder
    preprocessor = try ImagePreprocessor(budget: budget)
    imageBudget = budget
    self.dimensions = dimensions
    self.computeUnits = computeUnits
  }

  /// Downloads (or reuses), verifies and compiles the model, and loads the tokenizer. Model functions are loaded
  /// lazily on the first embedding that needs them, or up front with `prewarm`. `progress` is called on an
  /// arbitrary thread.
  public static func load(_ source: Source, budget: ImageBudget = .auto, dimensions: Int = 256,
                          store: ModelStore = ModelStore(), computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
                          progress: (@Sendable (InstallProgress) -> Void)? = nil) async throws -> EmbeddingGemma {
    guard (1...maxDimensions).contains(dimensions) else { throw EmbeddingGemmaError.invalidDimensions(dimensions) }
    let installed: InstalledModel
    switch source {
    case .huggingFace(let repo, let revision):
      let pinned = repo == defaultRepo && revision == defaultRevision ? defaultManifestSHA256 : nil
      installed = try await store.ensureModel(repo: repo, revision: revision, manifestSHA256: pinned,
                                              progress: progress)
    case .installed(let folder, let repo, let revision):
      installed = try await store.adopt(folder: folder, repo: repo, revision: revision)
    case .file(let folder):
      let files = folder.appending(path: FilesManifest.fileName), archived = folder.appending(path: ArchiveManifest.fileName)
      if !FileManager.default.fileExists(atPath: files.path), FileManager.default.fileExists(atPath: archived.path) {
        let bytes: Data
        do { bytes = try Data(contentsOf: archived) } catch { throw EmbeddingGemmaError.downloadFailed(underlying: error) }
        let revision = "file-" + Checksums.sha256(data: bytes).prefix(16)
        installed = try await store.ensureModel(archiveAt: folder, repo: "local", revision: revision,
                                                manifestSHA256: Checksums.sha256(data: bytes), progress: progress)
      } else {
        let bytes: Data
        do { bytes = try Data(contentsOf: files) } catch { throw EmbeddingGemmaError.downloadFailed(underlying: error) }
        let revision = "file-" + Checksums.sha256(data: bytes).prefix(16)  // hash and parse the same bytes
        installed = try await store.ensureModel(baseURL: folder, repo: "local", revision: revision, manifest: bytes,
                                                progress: progress)
      }
    }
    let encoder: TextEncoder
    do { encoder = try await TextEncoder(tokenizerFolder: installed.tokenizer) } catch let e as EmbeddingGemmaError {
      throw e
    } catch { throw EmbeddingGemmaError.modelLoadFailed(underlying: error) }
    return try EmbeddingGemma(installed: installed, encoder: encoder, budget: budget.resolved, dimensions: dimensions,
                              computeUnits: computeUnits)
  }

  /// Normalized, `dimensions` long. `orientation` is the EXIF orientation of `image` (photos often need it).
  public nonisolated func embed(image: CGImage, orientation: CGImagePropertyOrientation = .up) async throws -> [Float] {
    let preprocessor = preprocessor, boxed = SendableImage(image: image)
    let pre = try await Task.detached(priority: Task.currentPriority) {
      try preprocessor.preprocess(boxed.image, orientation: orientation)
    }.value
    try Task.checkCancellation()
    let name = pre.bucket.functionName
    return try CoreMLIO.truncateAndNormalize(try await function(name).embed(pre), dimensions: dimensions, function: name)
  }

  /// Adds the search-query prefix.
  public nonisolated func embed(query: String) async throws -> [Float] {
    try await embed(text: query, task: .query)
  }

  public nonisolated func embed(text: String, task: EmbeddingTask = .document) async throws -> [Float] {
    let tokens = try TextEncoder.pad(try encoder.encode(text: text, prompt: task.prompt))
    let name = tokens.functionName
    return try CoreMLIO.truncateAndNormalize(try await function(name).embed(tokens), dimensions: dimensions, function: name)
  }

  /// Loads functions now (default: the four text functions and the five image functions of `imageBudget`), so the
  /// first embeddings do not pay the load. Unknown names fail with `.modelLoadFailed`.
  public func prewarm(_ names: [String]? = nil) async throws {
    let all = names ?? (Self.textFunctions + Self.imageFunctions(budget: imageBudget))
    for name in all { _ = try await function(name) }
  }

  /// Releases every loaded function (they reload on next use), e.g. on a memory warning.
  public func unloadFunctions() {
    functions.removeAll()
  }

  /// The names of the functions loaded right now.
  public var loadedFunctions: [String] { functions.keys.sorted() }

  private func function(_ name: String) async throws -> LoadedFunction {
    if let task = functions[name] { return try await task.value }
    let config = MLModelConfiguration()
    config.functionName = name
    config.computeUnits = computeUnits
    let url = compiledModel
    let task = Task { () throws -> LoadedFunction in
      do { return LoadedFunction(model: try await MLModel.load(contentsOf: url, configuration: config)) } catch {
        throw EmbeddingGemmaError.modelLoadFailed(underlying: error)
      }
    }
    functions[name] = task
    do { return try await task.value } catch {
      functions[name] = nil  // let a later call retry
      throw error
    }
  }
}

/// A loaded Core ML function. `MLModel` is not marked `Sendable`, but its async `prediction(from:)` is documented as
/// safe to call concurrently (WWDC23 "Improve Core ML integration with async prediction"), and the model is never
/// mutated after loading. Inputs are built inside these methods from Sendable values, so no `MLMultiArray` crosses an
/// isolation boundary.
struct LoadedFunction: @unchecked Sendable {
  let model: MLModel

  /// The raw 768-d `embedding` for padded tokens (`input_ids`, `attention_mask`).
  func embed(_ tokens: PaddedTokens) async throws -> [Float] {
    let ids = try CoreMLIO.integers(tokens.ids, as: CoreMLIO.dataType(of: "input_ids", in: model, fallback: .int32),
                                    input: "input_ids")
    let mask = try CoreMLIO.integers(tokens.mask, as: CoreMLIO.dataType(of: "attention_mask", in: model, fallback: .int32),
                                     input: "attention_mask")
    return try await predict(["input_ids": ids, "attention_mask": mask])
  }

  /// The raw 768-d `embedding` for a bucket image (`pixels`).
  func embed(_ image: PreprocessedImage) async throws -> [Float] {
    let type = try CoreMLIO.dataType(of: "pixels", in: model, fallback: .float32)
    return try await predict(["pixels": image.multiArray(dataType: type)])
  }

  private func predict(_ inputs: [String: MLMultiArray]) async throws -> [Float] {
    let provider = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
    return try CoreMLIO.embedding(from: try await model.prediction(from: provider))
  }
}
