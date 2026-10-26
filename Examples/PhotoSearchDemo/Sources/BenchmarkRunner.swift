import Darwin
import EmbeddingGemma
import EmbeddingIndex
import Foundation
import Observation
import SwiftUI
import UIKit

/// The s6 benchmark (launch argument `-benchmark`): 1,000 photos at budget 70, then 200 at 280, plus text queries.
/// Writes `Documents/s6_device.json` (read by stages/s6_device/run.py) and shows "DONE".
///
/// Per-image time covers `embed(image:)` only (preprocessing + prediction), not the Photos fetch. Each budget loads
/// its own package when the bundle has one per budget (ADR-0010; `model_revisions` records each). Its
/// functions are loaded first (`prewarm`, recorded as `load_ms` and per function as `function_ms`). Peak memory is
/// `phys_footprint` (what iOS counts against the app's memory limit), sampled every 100 ms for the whole run. s6
/// launches it after the `-compile` run (`CompileRunner`, ADR-0009), so its loads are warm.
@MainActor @Observable
final class BenchmarkRunner {
  static let resultURL = URL.documentsDirectory.appending(path: "s6_device.json")
  static let plan: [(budget: EmbeddingGemma.ImageBudget, key: String, count: Int)] = [
    (.tokens70, "70", 1_000), (.tokens280, "280", 200),
  ]
  static let queries = [
    "a dog", "receipts", "screenshot of a chat", "sunset at the beach", "birthday cake", "a car", "food on a plate",
    "people smiling", "mountains", "a document with text", "a cat sleeping", "city at night", "flowers", "a selfie",
    "snow", "a whiteboard", "trees in a park", "a map", "a baby", "a concert",
  ]

  private(set) var status = "starting"
  private(set) var done = false

  func run(runID: String) async {
    UIApplication.shared.isIdleTimerDisabled = true
    defer { UIApplication.shared.isIdleTimerDisabled = false }
    try? FileManager.default.removeItem(at: Self.resultURL)
    let sampler = MemorySampler()
    sampler.start()
    var result = BenchmarkResult(runID: runID)
    do {
      try await measure(into: &result)
      result.status = "done"
    } catch {
      result.status = "error"
      result.error = String(describing: error)
    }
    result.peakRSSMB = Double(sampler.stop()) / 1_048_576
    result.thermalState = ProcessInfo.processInfo.thermalState.name
    result.finishedAt = Date().ISO8601Format()
    do {
      let enc = JSONEncoder()
      enc.outputFormatting = [.prettyPrinted, .sortedKeys]
      try enc.encode(result).write(to: Self.resultURL, options: .atomic)
      status = result.status == "done" ? "DONE" : "FAILED: \(result.error ?? "")"
    } catch {
      status = "FAILED to write the result: \(error)"
    }
    print("s6 benchmark \(status)")
    done = true
  }

  private func measure(into result: inout BenchmarkResult) async throws {
    status = "waiting for Photos access (tap Allow Full Access)"
    guard await PhotoLibrary.authorize() else { throw BenchmarkError.noPhotosAccess }
    let assets = PhotoLibrary.newestImages(limit: Self.plan.map(\.count).max() ?? 0)
    result.photosAvailable = assets.count
    guard !assets.isEmpty else { throw BenchmarkError.noPhotos }
    for step in Self.plan {
      status = "loading the model at budget \(step.key)"
      var phase = BudgetResult()
      let t0 = ContinuousClock.now
      var model: EmbeddingGemma? = try await DemoModel.load(budget: step.budget)
      for name in EmbeddingGemma.textFunctions + EmbeddingGemma.imageFunctions(budget: model!.imageBudget) {
        let f0 = ContinuousClock.now
        try await model!.prewarm([name])
        phase.functionMS[name] = (ContinuousClock.now - f0).ms
      }
      phase.loadMS = (ContinuousClock.now - t0).ms
      result.modelRevision = model!.modelRevision
      result.modelRevisions[step.key] = model!.modelRevision
      let index = try VectorIndex(url: FileManager.default.temporaryDirectory.appending(path: UUID().uuidString),
                                  dimensions: DemoModel.dimensions, modelRevision: model!.modelRevision)
      var times: [Double] = []
      for (i, asset) in assets.prefix(step.count).enumerated() {
        if i % 25 == 0 { status = "budget \(step.key): image \(i) of \(min(step.count, assets.count))" }
        guard let photo = await PhotoLibrary.photo(asset) else { phase.skipped += 1; continue }
        let start = ContinuousClock.now
        let v = try await model!.embed(image: photo.image, orientation: photo.orientation)
        times.append((ContinuousClock.now - start).ms)
        try index.upsert(id: photo.id, vector: v)
      }
      phase.images = times.count
      phase.msPerImage = Stats(times)
      phase.thermalStateAfter = ProcessInfo.processInfo.thermalState.name
      if step.key == "70" {
        status = "queries"
        var qtimes: [Double] = []
        for q in Self.queries {
          let start = ContinuousClock.now
          _ = index.search(try await model!.embed(query: q), k: 50)
          qtimes.append((ContinuousClock.now - start).ms)
        }
        result.query = QueryResult(queries: qtimes.count, ms: Stats(qtimes))
      }
      result.budgets[step.key] = phase
      model = nil  // release this budget's functions before loading the next
    }
  }
}

enum BenchmarkError: Error, CustomStringConvertible {
  case noPhotosAccess, noPhotos
  var description: String {
    switch self {
    case .noPhotosAccess: "Photos access was denied"
    case .noPhotos: "the photo library has no images"
    }
  }
}

/// The file stages/s6_device/run.py reads. Keys are snake_case; the device name is deliberately not recorded.
struct BenchmarkResult: Encodable {
  var runID: String
  var status = "running"
  var error: String?
  var deviceModel = machine()
  var osVersion = osVersionString()
  /// The last budget's `modelRevision` (kept for older readers; see `modelRevisions`).
  var modelRevision = ""
  /// `modelRevision` per budget step ("70", "280"): with one package per budget (ADR-0010) each has its own.
  var modelRevisions: [String: String] = [:]
  var photosAvailable = 0
  var budgets: [String: BudgetResult] = [:]
  var query = QueryResult(queries: 0, ms: Stats([]))
  var peakRSSMB = 0.0
  var thermalState = ""
  var startedAt = Date().ISO8601Format()
  var finishedAt = ""

  enum CodingKeys: String, CodingKey {
    case runID = "run_id", status, error, deviceModel = "device_model", osVersion = "os_version"
    case modelRevision = "model_revision", modelRevisions = "model_revisions", photosAvailable = "photos_available"
    case budgets, query
    case peakRSSMB = "peak_rss_mb", thermalState = "thermal_state", startedAt = "started_at", finishedAt = "finished_at"
  }

  init(runID: String) { self.runID = runID }
}

struct BudgetResult: Encodable {
  var images = 0
  var skipped = 0
  var loadMS = 0.0
  /// Load time per function. After the compile run (ADR-0009) every one should be a warm load.
  var functionMS: [String: Double] = [:]
  var msPerImage = Stats([])
  var thermalStateAfter = ""
  enum CodingKeys: String, CodingKey {
    case images, skipped, loadMS = "load_ms", functionMS = "function_ms", msPerImage = "ms_per_image", thermalStateAfter = "thermal_state_after"
  }
}

struct QueryResult: Encodable {
  var queries: Int
  var ms: Stats
}

/// Median and 90th percentile (nearest rank), plus the first value (cold) and the count.
struct Stats: Encodable {
  var median: Double?
  var p90: Double?
  var first: Double?
  var count: Int

  init(_ values: [Double]) {
    let s = values.sorted()
    func rank(_ q: Double) -> Double? { s.isEmpty ? nil : s[min(s.count - 1, Int((q * Double(s.count)).rounded(.up)) - 1)] }
    median = rank(0.5)
    p90 = rank(0.9)
    first = values.first
    count = values.count
  }
}

/// Peak `phys_footprint` over the run, sampled every 100 ms on a background task.
final class MemorySampler: @unchecked Sendable {
  private let lock = NSLock()
  private var peak: UInt64 = 0
  private var task: Task<Void, Never>?

  func start() {
    task = Task.detached(priority: .high) { [self] in
      while !Task.isCancelled {
        record()
        try? await Task.sleep(for: .milliseconds(100))
      }
    }
  }

  /// Stops sampling and returns the peak in bytes.
  func stop() -> UInt64 {
    task?.cancel()
    record()
    return lock.withLock { peak }
  }

  private func record() {
    let now = Self.physFootprint()
    lock.withLock { peak = max(peak, now) }
  }

  static func physFootprint() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    return kr == KERN_SUCCESS ? info.phys_footprint : 0
  }
}

func osVersionString() -> String {
  let v = ProcessInfo.processInfo.operatingSystemVersion
  return "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
}

/// `hw.machine`, e.g. "iPhone14,3".
func machine() -> String {
  var size = 0
  guard sysctlbyname("hw.machine", nil, &size, nil, 0) == 0, size > 0 else { return "unknown" }
  var buf = [CChar](repeating: 0, count: size)
  guard sysctlbyname("hw.machine", &buf, &size, nil, 0) == 0 else { return "unknown" }
  return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

extension Duration {
  var ms: Double { Double(components.seconds) * 1_000 + Double(components.attoseconds) / 1e15 }
}

extension ProcessInfo.ThermalState {
  var name: String {
    switch self {
    case .nominal: "nominal"
    case .fair: "fair"
    case .serious: "serious"
    case .critical: "critical"
    @unknown default: "unknown"
    }
  }
}

struct BenchmarkView: View {
  let runID: String
  @State private var runner = BenchmarkRunner()

  var body: some View {
    VStack(spacing: 16) {
      Text(runner.done ? runner.status : "Benchmark").font(.largeTitle.bold())
      if !runner.done { ProgressView(runner.status) }
      Text("run \(runID)").font(.caption.monospaced()).foregroundStyle(.secondary)
    }
    .padding()
    .task { await runner.run(runID: runID) }
  }
}
