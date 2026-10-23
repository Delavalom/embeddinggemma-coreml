// The published Core ML model. Rewritten by stages/s7_publish (Task 22) with the Hugging Face repo and the commit
// SHA of the upload; until then there is no published model and `.huggingFace` loads fail.
public extension EmbeddingGemma {
  static let defaultRepo = "pending/embeddinggemma-2-coreml"
  static let defaultRevision = "pending"
}
