import CoreML
import Foundation

func footprintMB() -> Int {
  var info = task_vm_info_data_t()
  var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
  let kr = withUnsafeMutablePointer(to: &info) {
    $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
  }
  return kr == KERN_SUCCESS ? Int(info.phys_footprint / 1_048_576) : -1
}
func peakMB() -> Int {
  var info = task_vm_info_data_t()
  var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
  let kr = withUnsafeMutablePointer(to: &info) {
    $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
  }
  return kr == KERN_SUCCESS ? Int(info.ledger_phys_footprint_peak / 1_048_576) : -1
}

// usage: probe <model.mlmodelc> <units ne|cpu|gpu> fn1,fn2,...   (loads one at a time, releases each; pauses for vmmap if PAUSE=1)
setvbuf(stdout, nil, _IOLBF, 0)
let a = CommandLine.arguments
let url = URL(fileURLWithPath: a[1])
let units: MLComputeUnits = a[2] == "cpu" ? .cpuOnly : a[2] == "gpu" ? .cpuAndGPU : .cpuAndNeuralEngine
print("pid \(getpid()) start footprint=\(footprintMB())MB")
for fn in a[3].split(separator: ",").map(String.init) {
  let c = MLModelConfiguration(); c.functionName = fn; c.computeUnits = units
  let t = Date()
  var m: MLModel? = try await MLModel.load(contentsOf: url, configuration: c)
  print("loaded \(fn) \(Int(Date().timeIntervalSince(t) * 1000))ms footprint=\(footprintMB())MB peak=\(peakMB())MB")
  if ProcessInfo.processInfo.environment["RELIEF"] == "1" {
    let freed = malloc_zone_pressure_relief(malloc_default_zone(), 0)
    print("relief freed=\(freed / 1_048_576)MB footprint=\(footprintMB())MB")
  }
  if let p = ProcessInfo.processInfo.environment["PAUSE_SECS"], let n = UInt32(p) { print("paused \(fn)"); sleep(n) }
  _ = m; m = nil
  print("released \(fn) footprint=\(footprintMB())MB")
}
