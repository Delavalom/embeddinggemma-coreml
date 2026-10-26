import CoreML
import Synchronization
import Testing
@testable import EmbeddingGemma

/// `FunctionLoad.withRetries`: the retry around a function load that may use the Neural Engine (see its doc).
@Suite struct FunctionLoadTests {
  struct Flaky: Error {}

  /// Fails the first `failures` calls, then returns the attempt number.
  final class Loader: Sendable {
    let failures: Int
    let calls = Mutex(0)
    init(failures: Int) { self.failures = failures }
    func load() throws -> Int {
      let n = calls.withLock { $0 += 1; return $0 }
      if n <= failures { throw Flaky() }
      return n
    }
  }

  @Test func neuralEngineLoadSucceedsAfterTwoFailures() async throws {
    let loader = Loader(failures: 2)
    let attempt = try await FunctionLoad.withRetries(computeUnits: .cpuAndNeuralEngine) { try loader.load() }
    #expect(attempt == 3)
  }

  @Test func neuralEngineLoadGivesUpAfterThreeAttempts() async {
    let loader = Loader(failures: 10)
    await #expect(throws: Flaky.self) {
      try await FunctionLoad.withRetries(computeUnits: .all) { try loader.load() }
    }
    #expect(loader.calls.withLock { $0 } == FunctionLoad.attempts)
  }

  @Test func cpuOnlyLoadIsNotRetried() async {
    let loader = Loader(failures: 1)
    await #expect(throws: Flaky.self) {
      try await FunctionLoad.withRetries(computeUnits: .cpuOnly) { try loader.load() }
    }
    #expect(loader.calls.withLock { $0 } == 1)
  }

  @Test func cancellationIsNotRetried() async {
    let calls = Mutex(0)
    await #expect(throws: CancellationError.self) {
      try await FunctionLoad.withRetries(computeUnits: .cpuAndNeuralEngine) { () throws -> Int in
        calls.withLock { $0 += 1 }
        throw CancellationError()
      }
    }
    #expect(calls.withLock { $0 } == 1)
  }
}
