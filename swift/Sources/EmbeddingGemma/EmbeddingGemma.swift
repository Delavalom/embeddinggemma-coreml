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
  /// Where the model comes from. The published model has one package per image budget (ADR-0010), and `load`
  /// installs only the one for the budget it resolves.
  public enum Source: Sendable {
    /// The published model at `revision` (a 40-character commit SHA) of `repo` (`owner/name`) on the Hugging Face
    /// Hub: `b<budget>/manifest.json` and the archive it names, installed in the store as `<revision>-b<budget>`.
    case huggingFace(repo: String = EmbeddingGemma.defaultRepo, revision: String)
    /// A local folder with the plain layout (`files.json`, `EmbeddingGemma2.mlpackage/`, `tokenizer/`), or with
    /// `manifest.json` and the archive it names: `folder/b<budget>/` when that folder exists (one package per
    /// budget), else `folder` itself (a package with every budget). Copied into the store (needs about 3x the files
    /// in free space).
    case file(URL)
    /// An unpacked, already verified layout (`EmbeddingGemma2.mlpackage/`, `tokenizer/`) of the budget being loaded,
    /// e.g. the app archive the caller downloaded, checked and extracted. Moved into the store as `repo`/`revision`,
    /// not copied: it must be on the store's volume and needs free space for the compiled model only
    /// (`ModelStore.adopt`).
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

  /// The Hugging Face commit SHA of the model (`file-<sha256 prefix of files.json>` of the folder used for `.file`).
  /// Index files store it, with `imageBudget`, so vectors from different models are never mixed.
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

  private init(installed: InstalledModel, revision: String, encoder: TextEncoder, budget: Int, dimensions: Int,
               computeUnits: MLComputeUnits) throws {
    modelRevision = revision
    compiledModel = installed.compiledModel
    lease = RevisionLease(installed.folder)
    self.encoder = encoder
    preprocessor = try ImagePreprocessor(budget: budget)
    imageBudget = budget
    self.dimensions = dimensions
    self.computeUnits = computeUnits
  }

  /// Resolves the image budget, then downloads (or reuses), verifies and compiles that budget's package, and loads
  /// the tokenizer (see `Source` for what each source installs). Model functions are loaded lazily on the first
  /// embedding that needs them, or up front with `prewarm`. `progress` is called on an arbitrary thread.
  public static func load(_ source: Source, budget: ImageBudget = .auto, dimensions: Int = 256,
                          store: ModelStore = ModelStore(), computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
                          progress: (@Sendable (InstallProgress) -> Void)? = nil) async throws -> EmbeddingGemma {
    guard (1...maxDimensions).contains(dimensions) else { throw EmbeddingGemmaError.invalidDimensions(dimensions) }
    let resolved = budget.resolved
    let (installed, revision) = try await install(source, budget: resolved, store: store, progress: progress)
    let encoder: TextEncoder
    do { encoder = try await TextEncoder(tokenizerFolder: installed.tokenizer) } catch let e as EmbeddingGemmaError {
      throw e
    } catch { throw EmbeddingGemmaError.modelLoadFailed(underlying: error) }
    return try EmbeddingGemma(installed: installed, revision: revision, encoder: encoder, budget: resolved,
                              dimensions: dimensions, computeUnits: computeUnits)
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

  /// Free space `prewarm` keeps by default: 3.5 GB of `volumeAvailableCapacity` (space not counting purgeable files).
  ///
  /// iOS 27.0.1 deleted the compiled Neural Engine files of our cache (`Library/Caches`, purgeable) whenever true free
  /// space fell below about 2 GB once other purgeable data was gone: 3 of 3 deliberate rounds, 0 of 4 at 2.25 GB or
  /// more, with the cleanup aiming for about 2.6 GB free; the next launch then compiled again, 45–120 s per function
  /// (docs/investigations/evidence/2026-10-27-ane-cache/e1-phone.md, E1e). A cold compile of one function writes about
  /// 0.5 GB (E1f: 4.2 GB for all nine plus the compiled model). So `prewarm` stops while a compile could still push the
  /// phone into that range by itself, which is how the cache was lost on 2026-10-27 (run 11).
  public static let prewarmReserve: Int64 = 3_500 * 1_048_576

  /// Loads functions now (default: the four text functions and the five image functions of `imageBudget`), so the
  /// first embeddings do not pay the load. Unknown names fail with `.modelLoadFailed`. Before each function, fails
  /// with `.insufficientStorage` if less than `keepingFree` bytes are free (see `prewarmReserve`); functions loaded
  /// so far stay loaded, and the rest still load on first use.
  public func prewarm(_ names: [String]? = nil, keepingFree reserve: Int64 = prewarmReserve) async throws {
    let all = names ?? (Self.textFunctions + Self.imageFunctions(budget: imageBudget))
    for name in all {
      if functions[name] == nil, let free = ModelStore.realFreeSpace(compiledModel), free < reserve {
        throw EmbeddingGemmaError.insufficientStorage(required: reserve, available: free)
      }
      _ = try await function(name)
    }
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
    let url = compiledModel, units = computeUnits
    let task = Task { () throws -> LoadedFunction in
      do {
        return LoadedFunction(model: try await FunctionLoad.withRetries(computeUnits: units) {
          try await MLModel.load(contentsOf: url, configuration: config)
        })
      } catch let e as CancellationError { throw e } catch {
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
