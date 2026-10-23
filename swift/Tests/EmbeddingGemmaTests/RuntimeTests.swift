import CoreML
import Foundation
import Testing
@testable import EmbeddingGemma

/// The actor against the contract model (stages/s5_swift/make_contract_model.py): real Core ML, the real function
/// names and I/O, and outputs we can recompute exactly. Skipped until that package is built.
@Suite(.enabled(if: TestSupport.contractPackage != nil,
                "build it: uv run --exact --group convert python -m stages.s5_swift.make_contract_model"))
struct RuntimeTests {
  /// W[k, d] = sin((k + 1) * (d + 1) / 100), as in make_contract_model.py.
  static func expected(_ features: [Double], dimensions: Int = 768) -> [Float] {
    let v = (0..<dimensions).map { d in
      features.enumerated().reduce(0.0) { $0 + $1.element * sin(Double($1.offset + 1) * Double(d + 1) / 100) }
    }
    let n = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
    return v.map { Float($0 / n) }
  }

  static func textFeatures(_ ids: [Int32]) -> [Double] {
    let p = TextEncoder.pad(ids)
    return [Double(ids.reduce(0) { $0 + Int($1) }) / 1e5, Double(ids.count) / 128, Double(p.length) / 128, 1]
  }

  static func load(budget: EmbeddingGemma.ImageBudget = .tokens70, dimensions: Int = 768) async throws -> EmbeddingGemma {
    let folder = try await TestSupport.publish(package: TestSupport.contractPackage!)
    return try await EmbeddingGemma.load(.file(folder), budget: budget, dimensions: dimensions,
                                         store: ModelStore(root: TestSupport.tempDir()), computeUnits: .cpuOnly)
  }

  static func close(_ a: [Float], _ b: [Float], tol: Float = 1e-4) -> Bool {
    a.count == b.count && zip(a, b).allSatisfy { abs($0 - $1) <= tol }
  }

  @Test func textUsesThePromptAndTheSmallestFunction() async throws {
    let m = try await Self.load(), enc = try await TokenizerCache.shared.encoder()
    let cases: [(String, EmbeddingTask)] = [
      ("receipts", .document), ("receipts", .classification), ("receipts", .clustering),
      (String(repeating: "a long caption about mountains and lakes ", count: 4), .document),  // > 32 tokens
      (String(repeating: "lorem ipsum dolor sit amet ", count: 60), .document),  // truncated at 128
    ]
    for (text, task) in cases {
      let got = try await m.embed(text: text, task: task)
      let ids = enc.encode(text: text, prompt: task.prompt)
      #expect(Self.close(got, Self.expected(Self.textFeatures(ids))), "\(task) \(ids.count) tokens")
    }
    let q = try await m.embed(query: "receipts")
    #expect(Self.close(q, Self.expected(Self.textFeatures(enc.encode(text: "receipts", prompt: EmbeddingTask.query.prompt)))))
    #expect(!Self.close(q, try await m.embed(text: "receipts")))  // the document prompt differs
  }

  @Test(arguments: [EmbeddingGemma.ImageBudget.tokens70, .tokens280])
  func imageUsesTheBucketFunction(budget: EmbeddingGemma.ImageBudget) async throws {
    let m = try await Self.load(budget: budget)
    #expect(m.imageBudget == budget.resolved)
    for (w, h) in [(240, 100), (100, 240), (300, 300), (400, 300), (300, 400)] {
      let img = try gradient(width: w, height: h)
      let pre = try ImagePreprocessor(budget: budget.resolved).preprocess(img)
      let n = pre.bucket.height * pre.bucket.width
      let means = (0..<3).map { c in
        Double(pre.planar[c * n..<(c + 1) * n].reduce(0) { $0 + Double(Float($1) * Float(1.0 / 255.0)) }) / Double(n)
      }
      let got = try await m.embed(image: img)
      #expect(Self.close(got, Self.expected(means + [Double(pre.bucket.height) / 1000])), "\(pre.bucket.functionName)")
    }
  }

  @Test func dimensionsTruncateThenNormalize() async throws {
    let full = try await Self.load(dimensions: 768), short = try await Self.load(dimensions: 256)
    let a = try await full.embed(query: "receipts"), b = try await short.embed(query: "receipts")
    #expect(a.count == 768 && b.count == 256)
    let head = Array(a.prefix(256)), n = head.reduce(0) { $0 + $1 * $1 }.squareRoot()
    #expect(Self.close(b, head.map { $0 / n }))
    #expect(abs(b.reduce(0) { $0 + $1 * $1 } - 1) < 1e-5)
  }

  @Test func fileSourceRevisionIsTheManifestHash() async throws {
    let folder = try await TestSupport.publish(package: TestSupport.contractPackage!)
    let store = ModelStore(root: TestSupport.tempDir())
    let m = try await EmbeddingGemma.load(.file(folder), store: store, computeUnits: .cpuOnly)
    let sha = try Checksums.sha256(file: folder.appending(path: FilesManifest.fileName))
    #expect(m.modelRevision == "file-" + sha.prefix(16))
    #expect(store.installed(revision: m.modelRevision) != nil)
  }
}
