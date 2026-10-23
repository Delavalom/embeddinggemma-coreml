public enum EmbeddingGemmaError: Error {
  case downloadFailed(underlying: any Error)
  case checksumMismatch(file: String)
  case unsupportedOS
  case modelLoadFailed(underlying: any Error)
  case revisionMismatch(expected: String, found: String)
  case invalidImage
  /// `dimensions` must be in 1...768 (the model's Matryoshka vector is truncated, then renormalized).
  case invalidDimensions(Int)
  /// The model's function lacks an expected input or output (e.g. a model that does not follow spec section 7).
  case contractViolation(String)
}
