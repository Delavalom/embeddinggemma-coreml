import CoreML
import SwiftUI
import UIKit

@main struct ProbeApp: App {
  var body: some Scene {
    WindowGroup {
      Text("probe").task {
        UIApplication.shared.isIdleTimerDisabled = true  // keeps the phone awake while the probe is in front
        await runProbe()
      }
    }
  }
}

func footprintMB() -> Int {
  var info = task_vm_info_data_t()
  var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
  let kr = withUnsafeMutablePointer(to: &info) {
    $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
  }
  return kr == KERN_SUCCESS ? Int(info.phys_footprint / 1_048_576) : -1
}

// Launch: -hold a,b,c   loads each with cpuAndNeuralEngine and keeps it; on failure retries once.
//         -tag X        names the output file Documents/probe-X.txt
func runProbe() async {
  let args = CommandLine.arguments
  func arg(_ k: String) -> String? { args.firstIndex(of: k).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
  let tag = arg("-tag") ?? "x"
  let out = URL.documentsDirectory.appending(path: "probe-\(tag).txt")
  var lines: [String] = []
  func log(_ s: String) {
    let l = "\(Date().formatted(date: .omitted, time: .standard)) \(s)"
    lines.append(l); try? lines.joined(separator: "\n").write(to: out, atomically: true, encoding: .utf8)
  }
  let root = URL.applicationSupportDirectory.appending(path: "EmbeddingGemma/local/file-8a7c99c73b81ba34/model.mlmodelc")
  let names = (arg("-hold") ?? "").split(separator: ",").map(String.init)
  log("start tag=\(tag) n=\(names.count) footprint=\(footprintMB())MB")
  let units: MLComputeUnits = switch arg("-units") { case "gpu": .cpuAndGPU; case "all": .all; case "cpu": .cpuOnly; default: .cpuAndNeuralEngine }
  log("units=\(arg("-units") ?? "ne") thermal=\(ProcessInfo.processInfo.thermalState.rawValue)")
  var held: [MLModel] = []
  for fn in names {
    for attempt in 1...3 {
      let c = MLModelConfiguration(); c.functionName = fn; c.computeUnits = units
      log("loading \(fn) attempt \(attempt)")
      let t = Date()
      do {
        held.append(try await MLModel.load(contentsOf: root, configuration: c))
        log("OK \(fn) \(Int(Date().timeIntervalSince(t) * 1000))ms held=\(held.count) footprint=\(footprintMB())MB")
        break
      } catch {
        log("FAIL \(fn) \(Int(Date().timeIntervalSince(t) * 1000))ms held=\(held.count) footprint=\(footprintMB())MB \((error as NSError).localizedDescription.prefix(50))")
      }
    }
  }
  log("holding \(held.count) footprint=\(footprintMB())MB")
  held.removeAll()
  log("released all footprint=\(footprintMB())MB")
  // -seq a,b,c: load each, log, release before the next
  for fn in (arg("-seq") ?? "").split(separator: ",").map(String.init) {
    for attempt in 1...2 {
      let c = MLModelConfiguration(); c.functionName = fn; c.computeUnits = .cpuAndNeuralEngine
      let t = Date()
      do {
        var m: MLModel? = try await MLModel.load(contentsOf: root, configuration: c)
        log("seq OK \(fn) \(Int(Date().timeIntervalSince(t) * 1000))ms loaded footprint=\(footprintMB())MB")
        _ = m; m = nil
        log("seq released \(fn) footprint=\(footprintMB())MB")
        break
      } catch { log("seq FAIL \(fn) attempt \(attempt) \(Int(Date().timeIntervalSince(t) * 1000))ms") }
    }
  }
  log("DONE")
}
