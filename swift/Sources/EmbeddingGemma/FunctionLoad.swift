import CoreML
import Foundation

/// Retries for loading one function.
///
/// On iOS 27.0.1 (iPhone 13 Pro Max) Core ML intermittently fails the first Neural Engine compile of a function,
/// 40–55 s in, with a misleading "`.functionName` property must be `nil` unless the model type is ML Program". On a
/// cool phone the same load succeeded on the next attempt in every case observed (5 failures in about 30 cold
/// compiles, with 0 to 5 other functions loaded). On a phone at thermal state "serious" after many compiles, three
/// immediate attempts all failed, twice. So loads that may use the Neural Engine get `attempts` tries, and before
/// each retry `pause` waits: `cooldown` sleeps a little longer each time, then until the phone is below "serious"
/// (at most `maxThermalWait`). `.cpuOnly` gets one try.
enum FunctionLoad {
  static let attempts = 3
  /// The first retry waits this long; each later one waits this much longer.
  static let backoff: Duration = .seconds(15)
  static let maxThermalWait: Duration = .seconds(300)
  static let thermalPoll: Duration = .seconds(10)

  static func withRetries<T>(computeUnits: MLComputeUnits, attempts: Int = attempts,
                             pause: (Int) async throws -> Void = cooldown,
                             _ load: () async throws -> T) async throws -> T {
    let tries = computeUnits == .cpuOnly ? 1 : max(1, attempts)
    var failure: any Error = CancellationError()
    for attempt in 1...tries {
      try Task.checkCancellation()
      do { return try await load() } catch let e as CancellationError { throw e } catch { failure = error }
      if attempt < tries { try await pause(attempt) }
    }
    throw failure
  }

  /// Waits `backoff * attempt`, then while the device is at thermal state `.serious` or worse, up to
  /// `maxThermalWait`.
  static func cooldown(after attempt: Int) async throws {
    try await Task.sleep(for: backoff * attempt)
    var waited: Duration = .zero
    while ProcessInfo.processInfo.thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue,
          waited < maxThermalWait {
      try await Task.sleep(for: thermalPoll)
      waited += thermalPoll
    }
  }
}

/// A loaded Core ML function. `MLModel` is not marked `Sendable`, but its async `prediction(from:)` is documented as
/// safe to call concurrently (WWDC23 "Improve Core ML integration with async prediction"), and the model is never
/// mutated after loading. Inputs are built inside these methods from Sendable values, so no `MLMultiArray` crosses an
/// isolation boundary.
struct LoadedFunction: @unchecked Sendable {
  let model: MLModel

  /// The raw 768-d `embedding` for padded tokens (`input_ids`, `attention_mask`).
  func embed(_ tokens: PaddedTokens) async throws -> [Float] {
    let ids = try CoreMLIO.integers(tokens.ids, as: CoreMLIO.dataType(of: "input_ids", in: model, fallback: .int32),
                                    input: "input_ids")
    let mask = try CoreMLIO.integers(tokens.mask, as: CoreMLIO.dataType(of: "attention_mask", in: model, fallback: .int32),
                                     input: "attention_mask")
    return try await predict(["input_ids": ids, "attention_mask": mask])
  }

  /// The raw 768-d `embedding` for a bucket image (`pixels`).
  func embed(_ image: PreprocessedImage) async throws -> [Float] {
    let type = try CoreMLIO.dataType(of: "pixels", in: model, fallback: .float32)
    return try await predict(["pixels": image.multiArray(dataType: type)])
  }

  private func predict(_ inputs: [String: MLMultiArray]) async throws -> [Float] {
    let provider = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
    return try CoreMLIO.embedding(from: try await model.prediction(from: provider))
  }
}
