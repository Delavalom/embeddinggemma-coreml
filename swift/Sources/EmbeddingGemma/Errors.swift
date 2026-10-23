public enum EmbeddingGemmaError: Error {
  case downloadFailed(underlying: any Error)
  case checksumMismatch(file: String)
  case unsupportedOS
  case modelLoadFailed(underlying: any Error)
  case revisionMismatch(expected: String, found: String)
  case invalidImage
}

/// Placeholder underlying error for the API skeleton; removed in Task 17.
struct NotImplementedError: Error, CustomStringConvertible {
  let description: String
  static let task17 = NotImplementedError(description: "not implemented until Task 17")
}
