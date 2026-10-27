import EmbeddingGemma
import Foundation
import Observation
import SwiftUI
import UIKit
import os

/// The s6 compile run (launch argument `-compile`, ADR-0009): right after an install, loads every function the
/// benchmark uses, one at a time, releasing each before the next, so each one compiles for this device once. Writes
/// `Documents/s6_compile.json` (read by stages/s6_device/run.py) and shows "DONE". Needs no Photos access.
///
/// Peak memory is `phys_footprint` sampled every 100 ms, as in the benchmark. The benchmark then runs in a new
/// process, so memory the allocator kept after a compile does not count against it.
@MainActor @Observable
final class CompileRunner {
  static let resultURL = URL.documentsDirectory.appending(path: "s6_compile.json")

  private(set) var status = "starting"
  private(set) var done = false

  func run(runID: String) async {
    UIApplication.shared.isIdleTimerDisabled = true
    defer { UIApplication.shared.isIdleTimerDisabled = false }
    try? FileManager.default.removeItem(at: Self.resultURL)
    var result = CompileResult(runID: runID)
    result.availableMemoryMBAtLaunch = Double(os_proc_available_memory()) / 1_048_576
    result.freeDiskMBAtLaunch = freeDiskMB()
    result.minFreeDiskMB = result.freeDiskMBAtLaunch
    let sampler = MemorySampler()
    sampler.start()
    do {
      try await compile(into: &result, sampler: sampler)
      result.status = "done"
    } catch {
      result.status = "error"
      result.error = String(describing: error)
    }
    result.peakRSSMB = Double(sampler.stop()) / 1_048_576
    result.thermalState = ProcessInfo.processInfo.thermalState.name
    result.finishedAt = Date().ISO8601Format()
    do {
      try Self.write(result)
      status = result.status == "done" ? "DONE" : "FAILED: \(result.error ?? "")"
    } catch {
      status = "FAILED to write the result: \(error)"
    }
    print("s6 compile \(status)")
    done = true
  }

  /// Every function of each budget's package. With one package per budget (ADR-0010) the text functions live in
  /// both packages and compile once in each; with one 14-function package (both budgets report the same revision)
  /// they compile once.
  /// After each function the result is written with status "running" (s6 waits for "done" or "error"), so a run that
  /// is stopped still shows how far it got, its peak so far and the thermal state.
  private func compile(into result: inout CompileResult, sampler: MemorySampler) async throws {
    var compiled = Set<String>()  // "<model revision>/<function>"
    for step in BenchmarkRunner.plan {
      status = "installing the model at budget \(step.key)"
      let model = try await DemoModel.load(budget: step.budget)
      result.modelRevision = model.modelRevision
      result.modelRevisions[step.key] = model.modelRevision
      let image = EmbeddingGemma.imageFunctions(budget: model.imageBudget)
      var stepMS: [String: Double] = [:]
      for name in EmbeddingGemma.textFunctions + image {
        guard compiled.insert("\(model.modelRevision)/\(name)").inserted else { continue }
        status = "compiling \(name) (budget \(step.key))"
        let t0 = ContinuousClock.now
        try await model.prewarm([name])
        let ms = (ContinuousClock.now - t0).ms
        stepMS[name] = ms
        if result.functionMS[name] == nil { result.functionMS[name] = ms }
        await model.unloadFunctions()
        result.peakRSSMB = Double(sampler.peak) / 1_048_576
        if let free = freeDiskMB() { result.minFreeDiskMB = min(result.minFreeDiskMB ?? free, free) }
        result.thermalState = ProcessInfo.processInfo.thermalState.name
        try? Self.write(result)
      }
      result.budgetFunctionMS[step.key] = stepMS
    }
  }
}

extension CompileRunner {
  static func write(_ result: CompileResult) throws {
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    try enc.encode(result).write(to: resultURL, options: .atomic)
  }
}

/// The file stages/s6_device/run.py reads after the compile run. Keys are snake_case.
struct CompileResult: Encodable {
  var runID: String
  var status = "running"
  var error: String?
  var deviceModel = machine()
  var osVersion = osVersionString()
  /// The last budget's `modelRevision` (kept for older readers; see `modelRevisions`).
  var modelRevision = ""
  /// `modelRevision` per budget step ("70", "280"): with one package per budget (ADR-0010) each has its own.
  var modelRevisions: [String: String] = [:]
  /// Load time per function name (its first load after the install, so its compile), from the first package
  /// that has it.
  var functionMS: [String: Double] = [:]
  /// Load time per function compiled at each budget step ("70", "280"): every function of that budget's package,
  /// minus those an earlier step already compiled in the same package.
  var budgetFunctionMS: [String: [String: Double]] = [:]
  /// `os_proc_available_memory()` at launch: how much more this app could use before iOS ends it.
  var availableMemoryMBAtLaunch = 0.0
  /// Free space for important use (`volumeAvailableCapacityForImportantUsage`) at launch and the lowest seen after
  /// each function: the Neural Engine compiler and its cache need disk.
  var freeDiskMBAtLaunch: Double?
  var minFreeDiskMB: Double?
  var peakRSSMB = 0.0
  var thermalState = ""
  var startedAt = Date().ISO8601Format()
  var finishedAt = ""

  enum CodingKeys: String, CodingKey {
    case runID = "run_id", status, error, deviceModel = "device_model", osVersion = "os_version"
    case modelRevision = "model_revision", modelRevisions = "model_revisions", functionMS = "function_ms"
    case budgetFunctionMS = "budget_function_ms"
    case availableMemoryMBAtLaunch = "available_memory_mb_at_launch", peakRSSMB = "peak_rss_mb"
    case freeDiskMBAtLaunch = "free_disk_mb_at_launch", minFreeDiskMB = "min_free_disk_mb"
    case thermalState = "thermal_state", startedAt = "started_at", finishedAt = "finished_at"
  }

  init(runID: String) { self.runID = runID }
}

struct CompileView: View {
  let runID: String
  @State private var runner = CompileRunner()

  var body: some View {
    VStack(spacing: 16) {
      Text(runner.done ? runner.status : "Compile").font(.largeTitle.bold())
      if !runner.done { ProgressView(runner.status) }
      Text("run \(runID)").font(.caption.monospaced()).foregroundStyle(.secondary)
    }
    .padding()
    .task { await runner.run(runID: runID) }
  }
}
