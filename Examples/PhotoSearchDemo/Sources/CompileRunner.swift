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
    let sampler = MemorySampler()
    sampler.start()
    do {
      try await compile(into: &result)
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
    print("s6 compile \(status)")
    done = true
  }

  private func compile(into result: inout CompileResult) async throws {
    for step in BenchmarkRunner.plan {
      status = "installing the model at budget \(step.key)"
      let model = try await DemoModel.load(budget: step.budget)
      result.modelRevision = model.modelRevision
      let image = EmbeddingGemma.imageFunctions(budget: model.imageBudget)
      for name in EmbeddingGemma.textFunctions + image where result.functionMS[name] == nil {
        status = "compiling \(name)"
        let t0 = ContinuousClock.now
        try await model.prewarm([name])
        result.functionMS[name] = (ContinuousClock.now - t0).ms
        await model.unloadFunctions()
      }
    }
  }
}

/// The file stages/s6_device/run.py reads after the compile run. Keys are snake_case.
struct CompileResult: Encodable {
  var runID: String
  var status = "running"
  var error: String?
  var deviceModel = machine()
  var osVersion = osVersionString()
  var modelRevision = ""
  /// Load time per function (its first load after the install, so its compile).
  var functionMS: [String: Double] = [:]
  /// `os_proc_available_memory()` at launch: how much more this app could use before iOS ends it.
  var availableMemoryMBAtLaunch = 0.0
  var peakRSSMB = 0.0
  var thermalState = ""
  var startedAt = Date().ISO8601Format()
  var finishedAt = ""

  enum CodingKeys: String, CodingKey {
    case runID = "run_id", status, error, deviceModel = "device_model", osVersion = "os_version"
    case modelRevision = "model_revision", functionMS = "function_ms"
    case availableMemoryMBAtLaunch = "available_memory_mb_at_launch", peakRSSMB = "peak_rss_mb"
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
