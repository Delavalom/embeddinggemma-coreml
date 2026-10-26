import Foundation

/// One package per image budget (ADR-0010): the published repo holds `b70/` and `b280/`, each with today's layout
/// (`manifest.json` + `EmbeddingGemma2.aar`, and the plain `files.json` layout), and a phone installs only the one
/// for its budget.
extension ModelStore {
  /// `b<budget>`: the folder of one image budget's package, in the published repo and in a local `.file` folder.
  public static func budgetFolder(_ budget: Int) -> String { "b\(budget)" }

  /// `<huggingFaceBase(repo, revision)>b<budget>/`, where that budget's `manifest.json` and archive are.
  public static func huggingFaceBase(repo: String, revision: String, budget: Int) -> URL {
    huggingFaceBase(repo: repo, revision: revision).appending(path: budgetFolder(budget) + "/")
  }

  /// `<revision>-b<budget>`: the store name (folder) of one budget's install of a Hugging Face commit.
  public static func installName(revision: String, budget: Int) -> String { "\(revision)-\(budgetFolder(budget))" }

  static func validate(budget: Int) throws {
    guard ArchiveManifest.budgets.contains(budget) else {
      throw EmbeddingGemmaError.invalidSource("image budget must be 70 or 280, got \(budget)")
    }
  }

  /// The image `budget` package (70 or 280) of `repo` at `revision` from the Hugging Face Hub:
  /// `b<budget>/manifest.json`, which must declare `"image_budget": <budget>`, and the one archive it names,
  /// installed as `<revision>-b<budget>` (`installName`). `manifestSHA256`, when given, pins that `manifest.json`.
  public func ensureModel(repo: String, revision: String, budget: Int, manifestSHA256: String? = nil,
                          progress: (@Sendable (InstallProgress) -> Void)? = nil) async throws -> InstalledModel {
    try Self.validate(repo: repo, revision: revision)
    try Self.validate(budget: budget)
    return try await ensureModel(archiveAt: Self.huggingFaceBase(repo: repo, revision: revision, budget: budget),
                                 repo: repo, revision: Self.installName(revision: revision, budget: budget),
                                 manifestSHA256: manifestSHA256,
                                 budget: BudgetRequirement(budget: budget, mustDeclare: true), progress: progress)
  }

  /// The installed image `budget` package of `repo` at `revision`, if its folder is complete.
  public func installed(repo: String, revision: String, budget: Int) -> InstalledModel? {
    installed(repo: repo, revision: Self.installName(revision: revision, budget: budget))
  }
}
