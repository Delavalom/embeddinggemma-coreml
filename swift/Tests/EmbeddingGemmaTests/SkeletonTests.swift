import Testing
@testable import EmbeddingGemma

@Test func taskPromptsMatchTheModelCard() {
  #expect(EmbeddingTask.query.prompt == "task: search result | query: ")
  #expect(EmbeddingTask.document.prompt == "title: none | text: ")
  #expect(EmbeddingTask.classification.prompt == "task: classification | query: ")
  #expect(EmbeddingTask.clustering.prompt == "task: clustering | query: ")
}

@Test(arguments: [0, -1, 769])
func invalidDimensionsAreRejectedBeforeAnyDownload(dimensions: Int) async {
  let store = ModelStore(root: TestSupport.tempDir(), fetcher: CountingFetcher())
  await #expect(throws: EmbeddingGemmaError.self) {
    _ = try await EmbeddingGemma.load(.huggingFace(revision: "r"), dimensions: dimensions, store: store)
  }
}

@Test func missingLocalFolderIsADownloadError() async {
  let error = await #expect(throws: EmbeddingGemmaError.self) {
    _ = try await EmbeddingGemma.load(.file(TestSupport.tempDir().appending(path: "nope")),
                                      store: ModelStore(root: TestSupport.tempDir()))
  }
  guard case .downloadFailed = error else { Issue.record("got \(String(describing: error))"); return }
}

/// Regression: our task enum must not shadow `Swift.Task` for importers.
@Test func swiftConcurrencyTaskIsNotShadowed() async throws {
  func check() throws { try Task.checkCancellation() }
  try check()
  let value = await Task { 41 + 1 }.value
  #expect(value == 42)
}
