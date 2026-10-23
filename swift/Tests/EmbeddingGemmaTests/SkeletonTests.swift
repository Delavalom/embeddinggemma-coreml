import Testing
@testable import EmbeddingGemma

@Test func taskPromptsMatchTheModelCard() {
  #expect(Task.query.prompt == "task: search result | query: ")
  #expect(Task.document.prompt == "title: none | text: ")
  #expect(Task.classification.prompt == "task: classification | query: ")
  #expect(Task.clustering.prompt == "task: clustering | query: ")
}

@Test func loadIsNotImplementedYet() async {
  await #expect(throws: EmbeddingGemmaError.self) {
    _ = try await EmbeddingGemma.load(.huggingFace(revision: EmbeddingGemma.defaultRevision))
  }
}
