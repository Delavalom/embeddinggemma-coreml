import Testing
@testable import EmbeddingGemma

@Suite struct TextEncoderTests {
  static func encoder() async throws -> TextEncoder { try await TokenizerCache.shared.encoder() }

  /// Every fixture text gives exactly s0's ids (prompt, bos and eos included).
  @Test func idsMatchS0ForAllFixtureTexts() async throws {
    let fixtures = try TestSupport.load(TextFixtures.self, "text_ids.json")
    #expect(fixtures.hf_revision == TestSupport.hfRevision)
    #expect(fixtures.texts.count == 200)
    let enc = try await Self.encoder()
    var mismatched: [String] = []
    for t in fixtures.texts {
      let task: EmbeddingTask = t.kind == "query" ? .query : .document
      #expect(t.kind == "query" || t.kind == "document")
      if try enc.encode(text: t.text, prompt: task.prompt) != t.ids { mismatched.append(t.id) }
    }
    #expect(mismatched.isEmpty, "ids differ for \(mismatched)")
  }

  @Test func paddingPicksTheSmallestFunction() throws {
    #expect(TextEncoder.lengths == [16, 32, 64, 128])
    for (n, want) in [(1, 16), (16, 16), (17, 32), (32, 32), (33, 64), (65, 128), (128, 128)] {
      #expect(TextEncoder.paddedLength(n) == want)
    }
    #expect(TextEncoder.paddedLength(129) == nil)
    #expect(throws: EmbeddingGemmaError.self) { try TextEncoder.pad(Array(repeating: 5, count: 129)) }
    let p = try TextEncoder.pad([2, 7, 1])
    #expect(p.ids == [2, 7, 1] + Array(repeating: 0, count: 13))
    #expect(p.mask == [1, 1, 1] + Array(repeating: 0, count: 13))
    #expect(p.tokenCount == 3 && p.functionName == "embed_text_16")
  }

  @Test func longTextIsTruncatedAt128KeepingBosAndEos() async throws {
    let enc = try await Self.encoder()
    let ids = try enc.encode(text: String(repeating: "lorem ipsum dolor sit amet ", count: 100), prompt: EmbeddingTask.document.prompt)
    #expect(ids.count == 128 && ids.first == 2 && ids.last == 1)
    let full = try enc.encode(text: String(repeating: "lorem ipsum dolor sit amet ", count: 100), prompt: "", maxTokens: .max)
    #expect(full.count > 128)
    let prompted = try enc.encode(text: String(repeating: "lorem ipsum dolor sit amet ", count: 100),
                                  prompt: EmbeddingTask.document.prompt, maxTokens: .max)
    #expect(Array(ids.dropLast()) == Array(prompted.prefix(127)))
    #expect(try TextEncoder.pad(ids).functionName == "embed_text_128")
  }

  /// Texts of 400 to 2,800 tokens, truncated by the reference tokenizer (`truncation=True, max_length=128`,
  /// stages/s5_swift/export_truncation.py): Swift gives the same 128 ids.
  @Test func truncationMatchesTheReferenceTokenizer() async throws {
    struct Row: Decodable { let id: String; let kind: String; let text: String; let ids: [Int32] }
    struct File: Decodable { let hf_revision: String; let max_tokens: Int; let texts: [Row] }
    let fx = try TestSupport.load(File.self, "truncation_ids.json")
    #expect(fx.hf_revision == TestSupport.hfRevision && fx.max_tokens == TextEncoder.maxTokens && fx.texts.count == 10)
    let enc = try await Self.encoder()
    for t in fx.texts {
      let task: EmbeddingTask = t.kind == "query" ? .query : .document
      #expect(try enc.encode(text: t.text, prompt: task.prompt) == t.ids, "\(t.id)")
    }
  }

  @Test func hugeInputIsCappedBeforeTokenizing() async throws {
    let enc = try await Self.encoder()
    let huge = String(repeating: "lorem ipsum dolor sit amet ", count: 200_000)  // 5.4 M characters
    let clock = ContinuousClock(), t0 = clock.now
    let ids = try enc.encode(text: huge, prompt: EmbeddingTask.document.prompt)
    #expect(ids.count == 128 && clock.now - t0 < .seconds(10))
    #expect(ids == (try enc.encode(text: String(huge.prefix(2_000)), prompt: EmbeddingTask.document.prompt)))
  }

  @Test func tokenizerSelfCheckMatchesS0() async throws {
    let fx = try TestSupport.load(TextFixtures.self, "text_ids.json")
    let t000 = try #require(fx.texts.first { $0.id == "t000" })
    #expect(TextEncoder.referenceIDs == t000.ids)
    #expect(TextEncoder.referenceText == EmbeddingTask.query.prompt + t000.text)
  }

  @Test func promptIsPrependedAsPlainText() async throws {
    let enc = try await Self.encoder()
    #expect(try enc.encode(text: "receipts", prompt: EmbeddingTask.query.prompt)
            == enc.encode(text: "task: search result | query: receipts", prompt: ""))
  }
}
