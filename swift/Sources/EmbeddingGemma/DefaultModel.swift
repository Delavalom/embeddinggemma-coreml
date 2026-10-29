// The published Core ML model. Written by stages/s7_publish (Task 22): the Hugging Face repo, the commit
// SHA of the upload and the sha256 of each image budget's manifest.json (b70/, b280/; ADR-0010).
public extension EmbeddingGemma {
  static let defaultRepo = "delavalom/embeddinggemma-2-coreml"
  static let defaultRevision = "8036f0bbba34dad5093108601ac6d060c63bbe76"
  /// sha256 of `b<budget>/manifest.json` at `defaultRevision`, per image budget. Loading the default model verifies
  /// the budget's `manifest.json` against it (on top of the pinned commit), so a compromised or rewritten Hub file
  /// is rejected.
  static let defaultManifestSHA256: [Int: String] = [70: "1edc9907eb54de411e6df7ccfbc610224884ff4229b90e9732671e034f011b4f", 280: "24f100644516fd6daba6fc57665b503baa59ac5e656896177d9e71c90a612cb5"]
}
