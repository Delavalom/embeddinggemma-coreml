import Foundation
import Synchronization
import Testing
@testable import EmbeddingGemma

/// Serves `https://huggingface.co/<repo>/resolve/<revision>/...` from a local folder and records every URL asked
/// for, so the Hugging Face path of the store runs offline.
final class HubFetcher: FileFetcher {
  let hub: URL
  let base: URL
  let urls = Mutex<[String]>([])
  init(hub: URL, repo: String, revision: String) {
    self.hub = hub
    base = ModelStore.huggingFaceBase(repo: repo, revision: revision)
  }
  func fetch(_ url: URL, to destination: URL, resumeFile: URL, progress: (@Sendable (Int64) -> Void)?) async throws {
    urls.withLock { $0.append(url.absoluteString) }
    guard url.absoluteString.hasPrefix(base.absoluteString) else { throw URLError(.fileDoesNotExist) }
    let local = hub.appending(path: String(url.absoluteString.dropFirst(base.absoluteString.count)))
    try await URLSessionFetcher().fetch(local, to: destination, resumeFile: resumeFile, progress: progress)
  }
}

/// Writes one budget's app download into `dir` (`manifest.json` + `EmbeddingGemma2.aar`); `imageBudget` nil
/// leaves `image_budget` out (a 14-function manifest).
func budgetArchive(in dir: URL, imageBudget: Int?) throws {
  let layout = try publishedFolder()
  try FileManager.default.removeItem(at: layout.appending(path: FilesManifest.fileName))
  try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
  let aar = dir.appending(path: "EmbeddingGemma2.aar")
  try ModelArchive.create(from: layout, to: aar)
  try writeManifest(dir, archive: aar, functions: ArchiveManifest.requiredFunctions(budget: imageBudget),
                    extra: imageBudget.map { ["image_budget": $0] } ?? [:])
}

func isBudgetMismatch(_ e: EmbeddingGemmaError, expected: Int, found: Int?) -> Bool {
  if case .downloadFailed(let u) = e { (u as? ManifestError) == .imageBudgetMismatch(expected: expected, found: found) }
  else { false }
}

@Suite struct BudgetManifestTests {
  static let sha = String(repeating: "ab", count: 20)

  @Test func imageBudgetNarrowsTheRequiredFunctions() throws {
    let only70 = ArchiveManifest.requiredFunctions(budget: 70)
    #expect(only70.count == 9 && only70.contains("embed_text_128") && only70.contains("embed_image_70_tall"))
    #expect(!only70.contains { $0.hasPrefix("embed_image_280") })
    #expect(ArchiveManifest.requiredFunctions(budget: nil) == ArchiveManifest.requiredFunctions)
    let m = try ArchiveManifest.decode(try manifestJSON(["image_budget": 70, "functions": only70]))
    #expect(m.imageBudget == 70)
    #expect(try ArchiveManifest.decode(try manifestJSON([:])).imageBudget == nil)
    #expect(throws: ManifestError.missing("embed_image_280_wide")) {  // without image_budget: all 14
      try ArchiveManifest.decode(try manifestJSON(["functions": ArchiveManifest.requiredFunctions(budget: 70)
        + ArchiveManifest.requiredFunctions(budget: 280).filter { $0 != "embed_image_280_wide" }]))
    }
    #expect(throws: ManifestError.missing("embed_image_280_tall")) {
      try ArchiveManifest.decode(try manifestJSON(["image_budget": 280, "functions": only70]))
    }
  }

  @Test(arguments: [0, 140, -70, 2800])
  func onlyBudgets70And280Exist(budget: Int) throws {
    #expect(throws: ManifestError.unsupportedImageBudget(budget)) {
      try ArchiveManifest.decode(try manifestJSON(["image_budget": budget]))
    }
  }

  @Test func checkServesBudget() throws {
    let m70 = try ArchiveManifest.decode(try manifestJSON(["image_budget": 70]))
    let all = try ArchiveManifest.decode(try manifestJSON([:]))
    try m70.check(serves: 70, mustDeclare: true)
    try all.check(serves: 280, mustDeclare: false)
    #expect(throws: ManifestError.imageBudgetMismatch(expected: 280, found: 70)) {
      try m70.check(serves: 280, mustDeclare: false)
    }
    #expect(throws: ManifestError.imageBudgetMismatch(expected: 70, found: nil)) {
      try all.check(serves: 70, mustDeclare: true)
    }
    let text = ManifestError.imageBudgetMismatch(expected: 280, found: 70).localizedDescription
    #expect(text.contains("image_budget 70") && text.contains("280"))
  }

  @Test func huggingFaceURLsAndInstallNamesArePerBudget() {
    #expect(ModelStore.huggingFaceBase(repo: "org/name", revision: Self.sha, budget: 70).absoluteString
            == "https://huggingface.co/org/name/resolve/\(Self.sha)/b70/")
    #expect(ModelStore.huggingFaceBase(repo: "org/name", revision: Self.sha, budget: 280)
      .appending(path: "manifest.json").absoluteString
            == "https://huggingface.co/org/name/resolve/\(Self.sha)/b280/manifest.json")
    #expect(ModelStore.installName(revision: Self.sha, budget: 280) == "\(Self.sha)-b280")
  }

  @Test func huggingFaceInstallsOnlyTheBudgetPackage() async throws {
    let hub = TestSupport.tempDir().appending(path: "hub"), root = TestSupport.tempDir()
    try budgetArchive(in: hub.appending(path: "b70"), imageBudget: 70)
    try budgetArchive(in: hub.appending(path: "b280"), imageBudget: 280)
    let fetcher = HubFetcher(hub: hub, repo: "org/name", revision: Self.sha)
    let s = ModelStore(root: root, fetcher: fetcher, compiler: FakeCompiler().compile, availableCapacity: { _ in nil })
    let (m, revision) = try await EmbeddingGemma.install(.huggingFace(repo: "org/name", revision: Self.sha), budget: 70,
                                                          store: s, progress: nil)
    let base = "https://huggingface.co/org/name/resolve/\(Self.sha)/b70/"
    #expect(fetcher.urls.withLock { $0 } == [base + "manifest.json", base + "EmbeddingGemma2.aar"])
    #expect(revision == Self.sha && m.revision == "\(Self.sha)-b70")
    #expect(m.folder.standardizedFileURL == root.appending(path: "org--name/\(Self.sha)-b70").standardizedFileURL)
    #expect(s.installed(repo: "org/name", revision: Self.sha, budget: 70) == m)
    #expect(s.installed(repo: "org/name", revision: Self.sha, budget: 280) == nil)
  }

  @Test(arguments: [280, nil] as [Int?])
  func huggingFaceRefusesAManifestOfAnotherBudget(declared: Int?) async throws {
    let hub = TestSupport.tempDir().appending(path: "hub")
    try budgetArchive(in: hub.appending(path: "b70"), imageBudget: declared)
    let fetcher = HubFetcher(hub: hub, repo: "org/name", revision: Self.sha)
    let s = ModelStore(root: TestSupport.tempDir(), fetcher: fetcher, compiler: FakeCompiler().compile,
                       availableCapacity: { _ in nil })
    await expectError({ _ = try await s.ensureModel(repo: "org/name", revision: Self.sha, budget: 70) }) {
      isBudgetMismatch($0, expected: 70, found: declared)
    }
    #expect(fetcher.urls.withLock { $0 }.count == 1)  // refused before the archive is downloaded
  }

  @Test(arguments: [0, 140])
  func huggingFaceRefusesUnknownBudgets(budget: Int) async {
    await expectError({ _ = try await store().ensureModel(repo: "org/name", revision: Self.sha, budget: budget) }) {
      if case .invalidSource = $0 { true } else { false }
    }
  }

  @Test func removeRevisionsKeepsExactlyTheInstallsPassed() async throws {
    let hub = TestSupport.tempDir().appending(path: "hub"), root = TestSupport.tempDir()
    try budgetArchive(in: hub.appending(path: "b70"), imageBudget: 70)
    try budgetArchive(in: hub.appending(path: "b280"), imageBudget: 280)
    let old = String(repeating: "cd", count: 20)
    func s(_ revision: String) -> ModelStore {
      ModelStore(root: root, fetcher: HubFetcher(hub: hub, repo: "org/name", revision: revision),
                 compiler: FakeCompiler().compile, availableCapacity: { _ in nil })
    }
    _ = try await s(old).ensureModel(repo: "org/name", revision: old, budget: 70)
    let new70 = try await s(Self.sha).ensureModel(repo: "org/name", revision: Self.sha, budget: 70)
    let new280 = try await s(Self.sha).ensureModel(repo: "org/name", revision: Self.sha, budget: 280)

    #expect(try await s(Self.sha).removeRevisions(except: [new70, new280]).map(\.lastPathComponent) == ["\(old)-b70"])
    #expect(try await s(Self.sha).removeRevisions(except: [new70]).map(\.lastPathComponent) == ["\(Self.sha)-b280"])
    #expect(s(Self.sha).installed(repo: "org/name", revision: Self.sha, budget: 70) == new70)
  }
}
