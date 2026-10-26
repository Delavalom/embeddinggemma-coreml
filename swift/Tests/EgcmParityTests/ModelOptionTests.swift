import Foundation
import Testing
@testable import egcm_parity

/// `--model` (ADR-0010): one bare path (a 14-function package for both budgets) or `70=<pkg>` and `280=<pkg>`.
/// stages/s5_swift/run.py passes exactly these forms.
@Suite struct ModelOptionTests {
  static func parse(_ models: [String]) throws -> Options {
    try Options.parse(models.flatMap { ["--model", $0] } + ["--tokenizer", "/t", "--fixtures", "/f", "--out", "/o.json"])
  }

  @Test func aBarePathServesBothBudgets() throws {
    let o = try Self.parse(["/m/EmbeddingGemma2.mlpackage"])
    #expect(o.model == .all(URL(filePath: "/m/EmbeddingGemma2.mlpackage")))
    #expect(o.model.package(budget: 70)?.path == "/m/EmbeddingGemma2.mlpackage")
    #expect(o.model.package(budget: 280)?.path == "/m/EmbeddingGemma2.mlpackage")
    #expect(o.model.argument == "/m/EmbeddingGemma2.mlpackage")
  }

  @Test(arguments: [["70=/a/p70.mlpackage", "280=/b/p280.mlpackage"], ["280=/b/p280.mlpackage", "70=/a/p70.mlpackage"]])
  func onePackagePerBudget(models: [String]) throws {
    let o = try Self.parse(models)
    #expect(o.model.package(budget: 70)?.path == "/a/p70.mlpackage")
    #expect(o.model.package(budget: 280)?.path == "/b/p280.mlpackage")
    #expect(o.model.argument == "70=/a/p70.mlpackage 280=/b/p280.mlpackage")
  }

  @Test(arguments: [
    [], ["70=/a"], ["280=/b"], ["70=/a", "70=/b", "280=/c"], ["140=/a", "280=/b"], ["/a", "70=/b", "280=/c"],
    ["/a", "/b"],
  ])
  func invalidModelArgumentsAreUsageErrors(models: [String]) {
    #expect(throws: ParityError.self) { _ = try Self.parse(models) }
  }

  @Test func perBudgetPackagesArePublishedIntoBudgetFolders() throws {
    let fm = FileManager.default
    let tmp = fm.temporaryDirectory.appending(path: "egcm-parity-\(UUID().uuidString)")
    defer { try? fm.removeItem(at: tmp) }
    let tok = tmp.appending(path: "tok")
    try fm.createDirectory(at: tok, withIntermediateDirectories: true)
    for name in ["tokenizer.json", "tokenizer_config.json"] { try Data("{}".utf8).write(to: tok.appending(path: name)) }
    var packages: [Int: URL] = [:]
    for b in [70, 280] {
      let pkg = tmp.appending(path: "p\(b)/EmbeddingGemma2.mlpackage")
      try fm.createDirectory(at: pkg, withIntermediateDirectories: true)
      try Data("\(b)".utf8).write(to: pkg.appending(path: "Manifest.json"))
      packages[b] = pkg
    }
    let root = try publish(.perBudget(packages), tokenizer: tok, into: tmp.appending(path: "published"))
    for b in [70, 280] {
      for path in ["files.json", "EmbeddingGemma2.mlpackage/Manifest.json", "tokenizer/tokenizer.json"] {
        #expect(fm.fileExists(atPath: root.appending(path: "b\(b)/\(path)").path), "b\(b)/\(path)")
      }
      let manifest = try String(contentsOf: root.appending(path: "b\(b)/EmbeddingGemma2.mlpackage/Manifest.json"),
                                encoding: .utf8)
      #expect(manifest == "\(b)")
    }
    #expect(!fm.fileExists(atPath: root.appending(path: "files.json").path))

    let flat = try publish(.all(packages[70]!), tokenizer: tok, into: tmp.appending(path: "flat"))
    #expect(fm.fileExists(atPath: flat.appending(path: "files.json").path))
    #expect(!fm.fileExists(atPath: flat.appending(path: "b70").path))
  }
}
