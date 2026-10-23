// The published Core ML model. Rewritten by stages/s7_publish (Task 22) with the Hugging Face repo, the commit SHA
// of the upload and the sha256 of its files.json; until then there is no published model and `.huggingFace` loads
// with these values fail.
public extension EmbeddingGemma {
  static let defaultRepo = "pending/embeddinggemma-2-coreml"
  static let defaultRevision = "pending"
  /// sha256 of `files.json` at `defaultRevision`. When set, loading the default model verifies `files.json` against
  /// it (on top of the pinned commit), so a compromised or rewritten Hub file is rejected.
  static let defaultFilesManifestSHA256: String? = nil
}
