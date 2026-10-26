import Foundation
import Testing
@testable import EmbeddingGemma

/// `.file(folder)` (ADR-0010): `folder/b<budget>/` when it exists, else `folder`; the revision is the hash of the
/// manifest of the folder actually used. Runs the install path with a fake compiler (no Core ML, no tokenizer).
@Suite struct FileSourceTests {
  static func install(_ root: URL, budget: Int, store s: ModelStore) async throws -> (InstalledModel, String) {
    let r = try await EmbeddingGemma.install(.file(root), budget: budget, store: s, progress: nil)
    return (r.installed, r.revision)
  }

  static func revision(of manifest: URL) throws -> String {
    "file-" + (try Checksums.sha256(file: manifest)).prefix(16)
  }

  /// `root/b70/` and `root/b280/`, each a plain layout with its own files.json (different bytes per budget).
  static func perBudgetFolder() throws -> URL {
    let root = TestSupport.tempDir().appending(path: "Model")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    for b in [70, 280] {
      try FileManager.default.moveItem(at: try publishedFolder(extra: ["tokenizer/budget.txt": "\(b)"]),
                                       to: root.appending(path: "b\(b)"))
    }
    return root
  }

  @Test func subfolderIsUsedOnlyWhenItIsADirectory() throws {
    let root = TestSupport.tempDir()
    #expect(EmbeddingGemma.fileFolder(root, budget: 70) == (root, false))
    try Data("not a folder".utf8).write(to: root.appending(path: "b70"))
    #expect(EmbeddingGemma.fileFolder(root, budget: 70) == (root, false))
    try FileManager.default.createDirectory(at: root.appending(path: "b280"), withIntermediateDirectories: true)
    let (folder, perBudget) = EmbeddingGemma.fileFolder(root, budget: 280)
    #expect(perBudget && folder.standardizedFileURL == root.appending(path: "b280").standardizedFileURL)
  }

  @Test func perBudgetFoldersInstallSeparatelyWithTheirOwnRevision() async throws {
    let root = try Self.perBudgetFolder(), s = store()
    let (m70, r70) = try await Self.install(root, budget: 70, store: s)
    let (m280, r280) = try await Self.install(root, budget: 280, store: s)
    #expect(r70 == (try Self.revision(of: root.appending(path: "b70/files.json"))))
    #expect(r280 == (try Self.revision(of: root.appending(path: "b280/files.json"))))
    #expect(r70 != r280 && m70.revision == r70 && m280.revision == r280 && m70.folder != m280.folder)
    let marker = try String(contentsOf: m280.tokenizer.appending(path: "budget.txt"), encoding: .utf8)
    #expect(marker == "280")
  }

  @Test func aFlatFolderServesEveryBudgetFromOneInstall() async throws {
    let root = try publishedFolder(), s = store()
    let (m70, r70) = try await Self.install(root, budget: 70, store: s)
    let (m280, r280) = try await Self.install(root, budget: 280, store: s)
    #expect(r70 == (try Self.revision(of: root.appending(path: "files.json"))))
    #expect(r70 == r280 && m70 == m280)
  }

  @Test func archiveFormWorksPerSubfolder() async throws {
    let root = TestSupport.tempDir().appending(path: "Model")
    try budgetArchive(in: root.appending(path: "b70"), imageBudget: 70)
    let (m, revision) = try await Self.install(root, budget: 70, store: store())
    #expect(revision == (try Self.revision(of: root.appending(path: "b70/manifest.json"))) && m.revision == revision)
  }

  @Test func aSubfolderManifestMustDeclareItsBudget() async throws {
    let root = TestSupport.tempDir().appending(path: "Model")
    try budgetArchive(in: root.appending(path: "b70"), imageBudget: nil)
    try budgetArchive(in: root.appending(path: "b280"), imageBudget: 70)
    await expectError({ _ = try await Self.install(root, budget: 70, store: store()) }) {
      isBudgetMismatch($0, expected: 70, found: nil)
    }
    await expectError({ _ = try await Self.install(root, budget: 280, store: store()) }) {
      isBudgetMismatch($0, expected: 280, found: 70)
    }
  }

  @Test func aFlatArchiveMayOmitTheBudgetButMustNotContradictIt() async throws {
    let all = TestSupport.tempDir().appending(path: "all"), only280 = TestSupport.tempDir().appending(path: "only280")
    try budgetArchive(in: all, imageBudget: nil)
    try budgetArchive(in: only280, imageBudget: 280)
    _ = try await Self.install(all, budget: 70, store: store())
    _ = try await Self.install(only280, budget: 280, store: store())
    await expectError({ _ = try await Self.install(only280, budget: 70, store: store()) }) {
      isBudgetMismatch($0, expected: 70, found: 280)
    }
  }
}
