import CoreGraphics
import Foundation

public actor EmbeddingGemma {
  public static let defaultRepo = "pending/embeddinggemma-2-coreml"
  public static let defaultRevision = "pending"

  public enum Source: Sendable {
    case huggingFace(repo: String = EmbeddingGemma.defaultRepo, revision: String)
    case file(URL)
  }

  public enum ImageBudget: Sendable { case auto, tokens70, tokens280 }

  private let revision: String
  private let dimensions: Int

  private init(revision: String, dimensions: Int) {
    self.revision = revision
    self.dimensions = dimensions
  }

  public static func load(_ _: Source, budget _: ImageBudget = .auto, dimensions _: Int = 256) async throws -> EmbeddingGemma {
    throw EmbeddingGemmaError.modelLoadFailed(underlying: NotImplementedError.task17)
  }

  /// Normalized, `dimensions` long.
  public func embed(image _: CGImage) async throws -> [Float] {
    throw EmbeddingGemmaError.modelLoadFailed(underlying: NotImplementedError.task17)
  }

  /// Adds the search-query prefix.
  public func embed(query _: String) async throws -> [Float] {
    throw EmbeddingGemmaError.modelLoadFailed(underlying: NotImplementedError.task17)
  }

  public func embed(text _: String, task _: EmbeddingTask = .document) async throws -> [Float] {
    throw EmbeddingGemmaError.modelLoadFailed(underlying: NotImplementedError.task17)
  }

  public nonisolated var modelRevision: String { revision }
}
