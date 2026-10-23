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
      if enc.encode(text: t.text, prompt: task.prompt) != t.ids { mismatched.append(t.id) }
    }
    #expect(mismatched.isEmpty, "ids differ for \(mismatched)")
  }

  @Test func paddingPicksTheSmallestFunction() {
    #expect(TextEncoder.lengths == [16, 32, 64, 128])
    for (n, want) in [(1, 16), (16, 16), (17, 32), (32, 32), (33, 64), (65, 128), (128, 128)] {
      #expect(TextEncoder.paddedLength(n) == want)
    }
    let p = TextEncoder.pad([2, 7, 1])
    #expect(p.ids == [2, 7, 1] + Array(repeating: 0, count: 13))
    #expect(p.mask == [1, 1, 1] + Array(repeating: 0, count: 13))
    #expect(p.tokenCount == 3 && p.functionName == "embed_text_16")
  }

  @Test func longTextIsTruncatedAt128KeepingBosAndEos() async throws {
    let enc = try await Self.encoder()
    let ids = enc.encode(text: String(repeating: "lorem ipsum dolor sit amet ", count: 100), prompt: EmbeddingTask.document.prompt)
    #expect(ids.count == 128 && ids.first == 2 && ids.last == 1)
    let full = enc.encode(text: String(repeating: "lorem ipsum dolor sit amet ", count: 100), prompt: "", maxTokens: .max)
    #expect(full.count > 128)
    let prompted = enc.encode(text: String(repeating: "lorem ipsum dolor sit amet ", count: 100),
                              prompt: EmbeddingTask.document.prompt, maxTokens: .max)
    #expect(Array(ids.dropLast()) == Array(prompted.prefix(127)))
    #expect(TextEncoder.pad(ids).functionName == "embed_text_128")
  }

  @Test func promptIsPrependedAsPlainText() async throws {
    let enc = try await Self.encoder()
    #expect(enc.encode(text: "receipts", prompt: EmbeddingTask.query.prompt)
            == enc.encode(text: "task: search result | query: receipts", prompt: ""))
  }
}
