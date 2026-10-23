/// Which prompt prefix EmbeddingGemma prepends to a text input.
public enum Task: Sendable {
  case query, document, classification, clustering

  /// The exact prefix the model was trained with.
  public var prompt: String {
    switch self {
    case .query: "task: search result | query: "
    case .document: "title: none | text: "
    case .classification: "task: classification | query: "
    case .clustering: "task: clustering | query: "
    }
  }
}
