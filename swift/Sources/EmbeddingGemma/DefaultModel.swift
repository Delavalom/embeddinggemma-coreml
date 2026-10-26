// The published Core ML model. Rewritten by stages/s7_publish (Task 22) with the Hugging Face repo, the commit SHA
// of the upload and the sha256 of each image budget's manifest.json (ADR-0010: `b70/manifest.json`,
// `b280/manifest.json`); until then there is no published model and `.huggingFace` loads with these values fail.
public extension EmbeddingGemma {
  static let defaultRepo = "pending/embeddinggemma-2-coreml"
  static let defaultRevision = "pending"
  /// sha256 of `b<budget>/manifest.json` at `defaultRevision`, keyed by image budget (70, 280). When set, loading
  /// the default model verifies that budget's `manifest.json` against it (on top of the pinned commit), so a
  /// compromised or rewritten Hub file is rejected. A budget without an entry is checked against the commit only.
  static let defaultManifestSHA256: [Int: String] = [:]
}
