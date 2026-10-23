import Testing
@testable import EmbeddingGemma

@Test func taskPromptsMatchTheModelCard() {
  #expect(EmbeddingTask.query.prompt == "task: search result | query: ")
  #expect(EmbeddingTask.document.prompt == "title: none | text: ")
  #expect(EmbeddingTask.classification.prompt == "task: classification | query: ")
  #expect(EmbeddingTask.clustering.prompt == "task: clustering | query: ")
}

@Test func loadIsNotImplementedYet() async {
  await #expect(throws: EmbeddingGemmaError.self) {
    _ = try await EmbeddingGemma.load(.huggingFace(revision: EmbeddingGemma.defaultRevision))
  }
}

/// Regression: our task enum must not shadow `Swift.Task` for importers.
@Test func swiftConcurrencyTaskIsNotShadowed() async throws {
  func check() throws { try Task.checkCancellation() }
  try check()
  let value = await Task { 41 + 1 }.value
  #expect(value == 42)
}
