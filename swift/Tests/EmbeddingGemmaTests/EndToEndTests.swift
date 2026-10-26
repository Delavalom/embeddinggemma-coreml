import Foundation
import Testing
@testable import EmbeddingGemma

struct EmbeddingFixtures: Decodable {
  struct Text: Decodable { let id: String; let kind: String; let text: String; let emb_b64: String }
  struct Image: Decodable { let id: String; let file: String; let budgets: [String: String] }
  let hf_revision: String
  let texts: [Text]
  let images: [Image]

  static func vector(_ b64: String) throws -> [Float] {
    let data = try #require(Data(base64Encoded: b64))
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }  // little-endian float32
  }
}

func cosine(_ a: [Float], _ b: [Float]) -> Float {
  let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
  return dot / (a.reduce(0) { $0 + $1 * $1 }.squareRoot() * b.reduce(0) { $0 + $1 * $1 }.squareRoot())
}

/// The real model from s4 against s0's vectors, on the Neural Engine. Skipped until stage 4 has run.
/// (Task 19's s5 stage runs the full fixture set; this is the unit-test sample.)
@Suite(.enabled(if: TestSupport.s4Package != nil || TestSupport.requireArtifacts,
                "artifacts/s4_compress has no .mlpackage: run just stage 4"),
       .serialized)
struct EndToEndTests {
  // The [s5_swift].min_cosine threshold, here against s0 on a sample of non-fragile images (the stage compares with
  // the FP reference on Swift's pixels, ADR-0008).
  static let minCosine: Float = 0.99

  @Test func textsMatchS0() async throws {
    let fx = try TestSupport.load(EmbeddingFixtures.self, "embeddings.json")
    let folder = try await TestSupport.publish(package: try #require(TestSupport.s4Package))
    let m = try await EmbeddingGemma.load(.file(folder), dimensions: 768, store: ModelStore(root: TestSupport.tempDir()))
    for t in fx.texts {
      let v = t.kind == "query" ? try await m.embed(query: t.text) : try await m.embed(text: t.text, task: .document)
      let c = cosine(v, try EmbeddingFixtures.vector(t.emb_b64))
      #expect(c >= Self.minCosine, "\(t.id): cosine \(c)")
    }
  }

  @Test(arguments: [70, 280])
  func imagesMatchS0(budget: Int) async throws {
    let fx = try TestSupport.load(EmbeddingFixtures.self, "embeddings.json")
    let folder = try await TestSupport.publish(package: try #require(TestSupport.s4Package))
    let m = try await EmbeddingGemma.load(.file(folder), budget: budget == 70 ? .tokens70 : .tokens280, dimensions: 768,
                                          store: ModelStore(root: TestSupport.tempDir()))
    for img in fx.images {
      let v = try await m.embed(image: try loadCGImage(TestSupport.fixturesDir.appending(path: img.file)))
      let c = cosine(v, try EmbeddingFixtures.vector(try #require(img.budgets[String(budget)])))
      #expect(c >= Self.minCosine, "\(img.id)@\(budget): cosine \(c)")
    }
  }
}
