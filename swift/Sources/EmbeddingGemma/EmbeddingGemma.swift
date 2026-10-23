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
public actor EmbeddingGemma {
  public enum Source: Sendable {
    /// The published model at `revision` (a commit SHA) of `repo` on the Hugging Face Hub.
    case huggingFace(repo: String = EmbeddingGemma.defaultRepo, revision: String)
    /// A local folder with the published layout: `files.json`, `EmbeddingGemma2.mlpackage/`, `tokenizer/`.
    case file(URL)
  }

  public enum ImageBudget: Sendable {
    /// 280 on A17 Pro and newer iPhones and on Macs, 70 otherwise (`DeviceTier`).
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

  /// The Hugging Face commit SHA of the model (`file-<sha256 prefix of files.json>` for `.file`). Index files store
  /// it so vectors from different models are never mixed.
  public nonisolated let modelRevision: String
  /// Soft tokens per image: 70 or 280.
  public nonisolated let imageBudget: Int
  /// Length of every returned vector.
  public nonisolated let dimensions: Int

  private let compiledModel: URL
  private let computeUnits: MLComputeUnits
  private let encoder: TextEncoder
  private let preprocessor: ImagePreprocessor
  /// One loaded function per name, loaded on first use (image functions only at `imageBudget`).
  private var functions: [String: Task<LoadedFunction, any Error>] = [:]

  private init(installed: InstalledModel, encoder: TextEncoder, budget: Int, dimensions: Int,
               computeUnits: MLComputeUnits) throws {
    modelRevision = installed.revision
    compiledModel = installed.compiledModel
    self.encoder = encoder
    preprocessor = try ImagePreprocessor(budget: budget)
    imageBudget = budget
    self.dimensions = dimensions
    self.computeUnits = computeUnits
  }

  /// Downloads (or reuses), verifies and compiles the model, and loads the tokenizer. Model functions are loaded
  /// lazily, on the first embedding that needs them.
  public static func load(_ source: Source, budget: ImageBudget = .auto, dimensions: Int = 256,
                          store: ModelStore = ModelStore(), computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
                          progress: (@Sendable (Double) -> Void)? = nil) async throws -> EmbeddingGemma {
    guard (1...maxDimensions).contains(dimensions) else { throw EmbeddingGemmaError.invalidDimensions(dimensions) }
    let installed: InstalledModel
    switch source {
    case .huggingFace(let repo, let revision):
      installed = try await store.ensureModel(repo: repo, revision: revision, progress: progress)
    case .file(let folder):
      let manifest = folder.appending(path: FilesManifest.fileName)
      let sha: String
      do { sha = try Checksums.sha256(file: manifest) } catch { throw EmbeddingGemmaError.downloadFailed(underlying: error) }
      installed = try await store.ensureModel(baseURL: folder, revision: "file-" + sha.prefix(16), progress: progress)
    }
    let encoder: TextEncoder
    do { encoder = try await TextEncoder(tokenizerFolder: installed.tokenizer) } catch {
      throw EmbeddingGemmaError.modelLoadFailed(underlying: error)
    }
    return try EmbeddingGemma(installed: installed, encoder: encoder, budget: budget.resolved, dimensions: dimensions,
                              computeUnits: computeUnits)
  }

  /// Normalized, `dimensions` long. `orientation` is the EXIF orientation of `image` (photos often need it).
  public func embed(image: CGImage, orientation: CGImagePropertyOrientation = .up) async throws -> [Float] {
    let pre = try preprocessor.preprocess(image, orientation: orientation)
    let embedding = try await function(pre.bucket.functionName).embed(pre)
    return CoreMLIO.truncateAndNormalize(embedding, dimensions: dimensions)
  }

  /// Adds the search-query prefix.
  public func embed(query: String) async throws -> [Float] {
    try await embed(text: query, task: .query)
  }

  public func embed(text: String, task: EmbeddingTask = .document) async throws -> [Float] {
    let tokens = TextEncoder.pad(encoder.encode(text: text, prompt: task.prompt))
    let embedding = try await function(tokens.functionName).embed(tokens)
    return CoreMLIO.truncateAndNormalize(embedding, dimensions: dimensions)
  }

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
    let ids = try CoreMLIO.integers(tokens.ids, as: CoreMLIO.dataType(of: "input_ids", in: model, fallback: .int32))
    let mask = try CoreMLIO.integers(tokens.mask, as: CoreMLIO.dataType(of: "attention_mask", in: model, fallback: .int32))
    return try await predict(["input_ids": ids, "attention_mask": mask])
  }

  /// The raw 768-d `embedding` for a bucket image (`pixels`).
  func embed(_ image: PreprocessedImage) async throws -> [Float] {
    let type = try CoreMLIO.dataType(of: "pixels", in: model, fallback: .float32)
    return try await predict(["pixels": image.multiArray(dataType: type == .float16 ? .float16 : .float32)])
  }

  private func predict(_ inputs: [String: MLMultiArray]) async throws -> [Float] {
    let provider = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
    return try CoreMLIO.embedding(from: try await model.prediction(from: provider))
  }
}
