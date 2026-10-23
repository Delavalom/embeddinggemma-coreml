import Foundation
import Synchronization
import Testing
@testable import EmbeddingGemma

/// Never answers, so a download only ends by cancellation.
final class HangingProtocol: URLProtocol, @unchecked Sendable {
  override class func canInit(with request: URLRequest) -> Bool { true }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  override func startLoading() {}
  override func stopLoading() {}
}

/// Answers every request with an HTTP status and no body.
final class StatusProtocol: URLProtocol, @unchecked Sendable {
  nonisolated(unsafe) static var status = 403
  override class func canInit(with request: URLRequest) -> Bool { true }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  override func startLoading() {
    let r = HTTPURLResponse(url: request.url!, statusCode: Self.status, httpVersion: "HTTP/1.1", headerFields: [:])!
    client?.urlProtocol(self, didReceive: r, cacheStoragePolicy: .notAllowed)
    client?.urlProtocol(self, didLoad: Data())
    client?.urlProtocolDidFinishLoading(self)
  }
  override func stopLoading() {}
}

func session(_ protocolClass: AnyClass) -> URLSession {
  let config = URLSessionConfiguration.ephemeral
  config.protocolClasses = [protocolClass]
  return URLSession(configuration: config)
}

@Suite struct DownloaderTests {
  /// A resumed attempt that fails with 403 (an expired signed URL), 410 or 416 drops the resume data and
  /// downloads from scratch, once.
  @Test(arguments: [403, 410, 416])
  func httpErrorOnResumeRetriesFromScratch(status: Int) async throws {
    let dir = TestSupport.tempDir(), resume = dir.appending(path: "f.resume")
    try Data("old resume data".utf8).write(to: resume)
    let attempts = Mutex<[Bool]>([])  // true = with resume data
    try await URLSessionFetcher.withResume(resume) { data in
      attempts.withLock { $0.append(data != nil) }
      if data != nil { throw HTTPStatusError(url: URL(string: "https://x")!, status: status) }
    }
    #expect(attempts.withLock { $0 } == [true, false])
    #expect(!FileManager.default.fileExists(atPath: resume.path))
  }

  @Test func cancellationOfAResumedAttemptKeepsTheResumeData() async throws {
    let dir = TestSupport.tempDir(), resume = dir.appending(path: "f.resume")
    try Data("resume".utf8).write(to: resume)
    await #expect(throws: CancellationError.self) {
      try await URLSessionFetcher.withResume(resume) { _ in throw CancellationError() }
    }
    #expect(FileManager.default.fileExists(atPath: resume.path))
  }

  @Test func staleResumeDataFallsBackToAFullDownload() async throws {
    let src = TestSupport.tempDir().appending(path: "f.bin"), dir = TestSupport.tempDir()
    try Data("payload".utf8).write(to: src)
    let dest = dir.appending(path: "f.bin"), resume = dir.appending(path: "f.bin.resume")
    try Data("not resume data".utf8).write(to: resume)
    let bytes = Mutex<Int64>(0)
    try await URLSessionFetcher().fetch(src, to: dest, resumeFile: resume) { b in bytes.withLock { $0 = max($0, b) } }
    #expect(try Data(contentsOf: dest) == Data("payload".utf8))
    #expect(!FileManager.default.fileExists(atPath: resume.path))
  }

  @Test func httpStatusIsAnError() async throws {
    StatusProtocol.status = 403
    let dir = TestSupport.tempDir()
    await #expect(throws: HTTPStatusError.self) {
      try await URLSessionFetcher(session: session(StatusProtocol.self))
        .fetch(URL(string: "https://example.invalid/f")!, to: dir.appending(path: "f"),
               resumeFile: dir.appending(path: "f.resume"), progress: nil)
    }
  }

  @Test(.timeLimit(.minutes(1))) func cancellingADownloadThrowsCancellationError() async throws {
    let dir = TestSupport.tempDir(), s = session(HangingProtocol.self)
    let task = Task {
      try await URLSessionFetcher(session: s).fetch(URL(string: "https://example.invalid/f")!, to: dir.appending(path: "f"),
                                                    resumeFile: dir.appending(path: "f.resume"), progress: nil)
    }
    try await Task.sleep(for: .milliseconds(200))
    task.cancel()
    await #expect(throws: CancellationError.self) { try await task.value }
  }

  /// A cancel that lands before the URLSession task exists must still cancel it (otherwise this would hang).
  @Test(.timeLimit(.minutes(1))) func cancelBeforeTheTaskStartsStillCancels() async throws {
    let dir = TestSupport.tempDir(), s = session(HangingProtocol.self)
    let task = Task {
      try await URLSessionFetcher(session: s).fetch(URL(string: "https://example.invalid/f")!, to: dir.appending(path: "f"),
                                                    resumeFile: dir.appending(path: "f.resume"), progress: nil)
    }
    task.cancel()
    await #expect(throws: CancellationError.self) { try await task.value }
  }
}
