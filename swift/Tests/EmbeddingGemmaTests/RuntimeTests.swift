import CoreML
import Foundation
import Testing
@testable import EmbeddingGemma

/// The actor against the contract model (stages/s5_swift/make_contract_model.py): real Core ML, the real function
/// names and I/O, and outputs we can recompute exactly. Skipped until that package is built.
@Suite(.enabled(if: TestSupport.contractPackage != nil || TestSupport.requireArtifacts,
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

  static func textFeatures(_ ids: [Int32]) throws -> [Double] {
    let p = try TextEncoder.pad(ids)
    return [Double(ids.reduce(0) { $0 + Int($1) }) / 1e5, Double(ids.count) / 128, Double(p.length) / 128, 1]
  }

  static func load(budget: EmbeddingGemma.ImageBudget = .tokens70, dimensions: Int = 768) async throws -> EmbeddingGemma {
    let folder = try await TestSupport.publish(package: try #require(TestSupport.contractPackage))
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
      let ids = try enc.encode(text: text, prompt: task.prompt)
      #expect(Self.close(got, Self.expected(try Self.textFeatures(ids))), "\(task) \(ids.count) tokens")
    }
    let q = try await m.embed(query: "receipts")
    #expect(Self.close(q, Self.expected(try Self.textFeatures(enc.encode(text: "receipts", prompt: EmbeddingTask.query.prompt)))))
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
    let folder = try await TestSupport.publish(package: try #require(TestSupport.contractPackage))
    let store = ModelStore(root: TestSupport.tempDir())
    let m = try await EmbeddingGemma.load(.file(folder), store: store, computeUnits: .cpuOnly)
    let sha = try Checksums.sha256(file: folder.appending(path: FilesManifest.fileName))
    #expect(m.modelRevision == "file-" + sha.prefix(16))
    #expect(store.installed(repo: "local", revision: m.modelRevision) != nil)
  }

  /// The extra contract functions, called directly: NaN output throws, FP16 and [1, 768, 1, 1] outputs read right.
  static func loadFunction(_ name: String) async throws -> LoadedFunction {
    let folder = try await TestSupport.publish(package: try #require(TestSupport.contractPackage))
    let m = try await ModelStore(root: TestSupport.tempDir()).ensureModel(baseURL: folder, revision: "contract")
    let config = MLModelConfiguration()
    config.functionName = name
    config.computeUnits = .cpuOnly
    return LoadedFunction(model: try await MLModel.load(contentsOf: m.compiledModel, configuration: config))
  }

  @Test func outputVariantsAreReadOrRejected() async throws {
    let enc = try await TokenizerCache.shared.encoder()
    let ids = try enc.encode(text: "receipts", prompt: EmbeddingTask.query.prompt)
    let tokens = try TextEncoder.pad(ids), want = Self.expected(try Self.textFeatures(ids))
    for variant in ["fp16", "4d"] {
      let raw = try await Self.loadFunction("test_\(variant)_text_16").embed(tokens)
      let got = try CoreMLIO.truncateAndNormalize(raw, dimensions: 768, function: variant)
      #expect(Self.close(got, want, tol: variant == "fp16" ? 2e-3 : 1e-4), "\(variant)")
    }
    let nan = try await Self.loadFunction("test_nan_text_16").embed(tokens)
    #expect(throws: EmbeddingGemmaError.self) { try CoreMLIO.truncateAndNormalize(nan, dimensions: 256, function: "test_nan_text_16") }
    do {
      _ = try CoreMLIO.truncateAndNormalize(nan, dimensions: 256, function: "test_nan_text_16")
    } catch EmbeddingGemmaError.nonFiniteOutput(let f) { #expect(f == "test_nan_text_16") }
  }

  @Test func prewarmLoadsNineFunctionsAndUnloadReleasesThem() async throws {
    let m = try await Self.load(budget: .tokens70)
    #expect(await m.loadedFunctions.isEmpty)
    try await m.prewarm(keepingFree: 0)  // independent of this machine's free space
    #expect(await m.loadedFunctions == (EmbeddingGemma.textFunctions + EmbeddingGemma.imageFunctions(budget: 70)).sorted())
    #expect(await m.loadedFunctions.count == 9)
    await m.unloadFunctions()
    #expect(await m.loadedFunctions.isEmpty)
    _ = try await m.embed(query: "receipts")  // reloads on demand
    #expect(await m.loadedFunctions == ["embed_text_16"])
    await #expect(throws: EmbeddingGemmaError.self) { try await m.prewarm(["no_such_function"]) }
  }

  /// Below the reserve, `prewarm` loads nothing new and says how much is free; functions already loaded pass, and
  /// the rest still load on first use.
  @Test func prewarmStopsBelowTheFreeSpaceReserve() async throws {
    let m = try await Self.load(budget: .tokens70)
    do {
      try await m.prewarm(keepingFree: .max)
      Issue.record("expected insufficientStorage")
    } catch EmbeddingGemmaError.insufficientStorage(let required, let available) {
      #expect(required == .max)
      #expect(available > 0)
    }
    #expect(await m.loadedFunctions.isEmpty)
    _ = try await m.embed(query: "receipts")
    try await m.prewarm(["embed_text_16"], keepingFree: .max)
    #expect(await m.loadedFunctions == ["embed_text_16"])
  }

  /// Queries are not queued behind image preprocessing: both run at once and give the same vectors as alone.
  @Test func concurrentTextAndImageCalls() async throws {
    let m = try await Self.load(budget: .tokens280)
    let img = try TestSupport.synthetic(height: 3024, width: 4032)
    let q0 = try await m.embed(query: "receipts"), i0 = try await m.embed(image: img)
    async let i1 = m.embed(image: img)
    async let qs = withThrowingTaskGroup(of: [Float].self) { g in
      for _ in 0..<8 { g.addTask { try await m.embed(query: "receipts") } }
      return try await g.reduce(into: [[Float]]()) { $0.append($1) }
    }
    let (img1, queries) = try await (i1, qs)
    #expect(img1 == i0 && queries.allSatisfy { $0 == q0 })
  }

  @Test func revisionFolderIsLeasedWhileTheModelLives() async throws {
    let folder = try await TestSupport.publish(package: try #require(TestSupport.contractPackage))
    let root = TestSupport.tempDir(), store = ModelStore(root: root)
    var m: EmbeddingGemma? = try await EmbeddingGemma.load(.file(folder), store: store, computeUnits: .cpuOnly)
    #expect(try await store.removeRevisions().isEmpty)
    let revision = m!.modelRevision
    m = nil
    #expect(try await store.removeRevisions().count == 1)
    #expect(store.installed(repo: "local", revision: revision) == nil)
  }
}
