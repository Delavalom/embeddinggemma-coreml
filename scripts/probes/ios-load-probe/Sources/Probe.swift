import CoreML
import SwiftUI
import UIKit
import os

// The ANE cache probe (docs/superpowers/plans/2026-10-27-ios27-ane-cache-investigation.md, Task 3). One variable per
// launch; every launch writes Documents/probe-<tag>.json with its environment and every load.
//
// Launch arguments:
//   -tag X              output Documents/probe-X.json
//   -model b70|t16      bundled package (default b70); compiled once on the device into Application Support/probe/
//   -fns a,b            functions to load, one at a time, each released before the next
//   -noFunctionName     load without config.functionName (the 1-function package t16)
//   -units ne|gpu|cpu|all
//   -api async|sync     MLModel.load(contentsOf:configuration:) (default) or MLModel(contentsOf:configuration:)
//   -from as|tmp|group  load from Application Support (default, like ModelStore), from a fresh tmp copy, or from a
//                       copy made once in the App Group container (same path and inodes across app updates)
//   -retries N          tries per function (default 1: the experiments want the raw failure)
//   -wipePayload        delete every regular file under our E5 cache's *.bundle folders first (H6), keep folders
//   -fill MB            append MB of non-purgeable data to Documents/filler.bin (E1e: low free space, one variable)
//   -unfill             delete Documents/filler.bin

@main struct ProbeApp: App {
  @State private var status = "probe"
  var body: some Scene {
    WindowGroup {
      Text(status).font(.title.monospaced()).task {
        UIApplication.shared.isIdleTimerDisabled = true  // the probe must stay in front and awake
        status = await runProbe()
      }
    }
  }
}

struct Load: Codable {
  var fn: String
  var attempt = 1
  var ok = false
  var ms = 0
  var error: String?
  var footprintMB = 0
  var peakMB = 0
  var thermal = 0
  var freeDiskMB = 0
  var note: String?
}

struct Report: Codable {
  var tag: String
  var args: [String]
  var osBuild: String
  var startedAt: String
  var finishedAt = ""
  var secondsSinceLastProbe: Double?
  var thermalStart: Int
  var thermalEnd = 0
  var freeDiskMBStart: Int
  var freeDiskMBEnd = 0
  var availMemMBStart: Int
  var modelPath = ""
  var modelFiles: [String: [String: String]] = [:]  // file → size, mtime, inode
  var compileMS: Int?
  var loads: [Load] = []
  var status = "running"
}

func footprint() -> (Int, Int) {
  var info = task_vm_info_data_t()
  var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
  let kr = withUnsafeMutablePointer(to: &info) {
    $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
      task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
    }
  }
  guard kr == KERN_SUCCESS else { return (-1, -1) }
  return (Int(info.phys_footprint / 1_048_576), Int(info.ledger_phys_footprint_peak / 1_048_576))
}

func freeDiskMB() -> Int {
  let v = try? URL.documentsDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
  return Int((v?.volumeAvailableCapacityForImportantUsage ?? -1_048_576) / 1_048_576)
}

func osBuild() -> String {
  var size = 0
  sysctlbyname("kern.osversion", nil, &size, nil, 0)
  var b = [CChar](repeating: 0, count: size)
  sysctlbyname("kern.osversion", &b, &size, nil, 0)
  return String(cString: b)
}

func fileFacts(_ dir: URL) -> [String: [String: String]] {
  var out: [String: [String: String]] = [:]
  for name in ["model.mil", "weights/weight.bin", "coremldata.bin"] {
    guard let a = try? FileManager.default.attributesOfItem(atPath: dir.appending(path: name).path) else { continue }
    out[name] = ["size": "\(a[.size] ?? "")",
                 "mtime": "\((a[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)",
                 "inode": "\(a[.systemFileNumber] ?? "")"]
  }
  return out
}

/// Deletes the regular files under our own E5 cache's `*.bundle` folders and keeps the folders, which is the state
/// iOS's purge left behind on 2026-10-27 (evidence/2026-10-27-ane-cache/e2-run11-ios-purge.md). Returns the count.
func wipePayload() -> Int {
  let root = URL.cachesDirectory.appending(path: "\(Bundle.main.bundleIdentifier!)/com.apple.e5rt.e5bundlecache")
  guard let it = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey]) else {
    return 0
  }
  var removed = 0
  for case let f as URL in it where f.path.contains(".bundle/") {
    if (try? f.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true {
      if (try? FileManager.default.removeItem(at: f)) != nil { removed += 1 }
    }
  }
  return removed
}

let fillerURL = URL.documentsDirectory.appending(path: "filler.bin")

/// Appends `mb` MB to Documents/filler.bin in 64 MB writes (real blocks: random bytes, so nothing can be shared or
/// compressed), excluded from backup. Returns the MB written; stops early if a write fails (disk full).
func fill(mb: Int) -> Int {
  if !FileManager.default.fileExists(atPath: fillerURL.path) {
    FileManager.default.createFile(atPath: fillerURL.path, contents: nil)
    var v = URLResourceValues()
    v.isExcludedFromBackup = true
    var u = fillerURL
    try? u.setResourceValues(v)
  }
  guard let h = try? FileHandle(forWritingTo: fillerURL) else { return 0 }
  defer { try? h.close() }
  _ = try? h.seekToEnd()
  var chunk = Data(count: 64 << 20)
  chunk.withUnsafeMutableBytes { arc4random_buf($0.baseAddress!, $0.count) }
  var written = 0
  while written + 64 <= mb {
    do { try h.write(contentsOf: chunk) } catch { break }
    written += 64
  }
  try? h.synchronize()
  return written
}

/// Compiles the bundled package once into Application Support/probe/<model>.mlmodelc (like ModelStore) and reuses it
/// on later launches. Returns the compiled URL and the compile time if it compiled now.
func compiledModel(_ name: String) async throws -> (URL, Int?) {
  let dir = URL.applicationSupportDirectory.appending(path: "probe")
  let compiled = dir.appending(path: "\(name).mlmodelc")
  if FileManager.default.fileExists(atPath: compiled.path) { return (compiled, nil) }
  guard let src = Bundle.main.url(forResource: name, withExtension: "mlpackage", subdirectory: "ProbeModels") else {
    throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: "ProbeModels/\(name).mlpackage"])
  }
  let t = Date()
  let tmp = try await MLModel.compileModel(at: src)
  try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
  try FileManager.default.moveItem(at: tmp, to: compiled)
  return (compiled, Int(Date().timeIntervalSince(t) * 1000))
}

func runProbe() async -> String {
  let args = CommandLine.arguments
  func arg(_ k: String) -> String? { args.firstIndex(of: k).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
  let tag = arg("-tag") ?? "x"
  let out = URL.documentsDirectory.appending(path: "probe-\(tag).json")
  let stamp = URL.documentsDirectory.appending(path: "last-probe-end.txt")
  let last = (try? String(contentsOf: stamp, encoding: .utf8)).flatMap(Double.init)
  var r = Report(tag: tag, args: Array(args.dropFirst()), osBuild: osBuild(), startedAt: Date().ISO8601Format(),
                 secondsSinceLastProbe: last.map { Date().timeIntervalSince1970 - $0 },
                 thermalStart: ProcessInfo.processInfo.thermalState.rawValue, freeDiskMBStart: freeDiskMB(),
                 availMemMBStart: Int(os_proc_available_memory() / 1_048_576))
  func save() { if let d = try? JSONEncoder().encode(r) { try? d.write(to: out, options: .atomic) } }
  save()
  if args.contains("-wipePayload") {
    r.loads.append(Load(fn: "wipePayload", ok: true, freeDiskMB: freeDiskMB(), note: "removed \(wipePayload()) files"))
    save()
  }
  if let mb = arg("-fill").flatMap(Int.init) {
    let before = freeDiskMB()
    let n = fill(mb: mb)
    r.loads.append(Load(fn: "fill", ok: n >= mb - 64, ms: n, freeDiskMB: freeDiskMB(), note: "wrote \(n) MB, app free \(before)→\(freeDiskMB())"))
    save()
  }
  if args.contains("-unfill") {
    let size = (try? FileManager.default.attributesOfItem(atPath: fillerURL.path)[.size] as? Int) ?? 0
    let ok = (try? FileManager.default.removeItem(at: fillerURL)) != nil
    r.loads.append(Load(fn: "unfill", ok: ok, ms: size / 1_048_576, freeDiskMB: freeDiskMB(), note: "removed \(size / 1_048_576) MB"))
    save()
  }
  let fns = (arg("-fns") ?? "").split(separator: ",").map(String.init)
  if fns.isEmpty && !args.contains("-model") {  // a fill/unfill/wipe-only launch: don't compile the package
    r.status = "done"
    r.thermalEnd = ProcessInfo.processInfo.thermalState.rawValue
    r.freeDiskMBEnd = freeDiskMB()
    r.finishedAt = Date().ISO8601Format()
    save()
    return "DONE \(r.status)"
  }
  do {
    let (compiled, compileMS) = try await compiledModel(arg("-model") ?? "b70")
    r.compileMS = compileMS
    var url = compiled
    if arg("-from") == "tmp" {  // H4: same bytes, a new path every launch
      url = FileManager.default.temporaryDirectory.appending(path: "\(UUID().uuidString).mlmodelc")
      try FileManager.default.copyItem(at: compiled, to: url)
    }
    if arg("-from") == "group" {  // E1g: a path outside the app container
      guard let g = FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: "group.com.delavalom.egcm.demo")
      else { throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: "app group container"]) }
      url = g.appending(path: "probe/\(compiled.lastPathComponent)")
      if !FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: compiled, to: url)
      }
    }
    r.modelPath = url.path
    r.modelFiles = fileFacts(url)
    save()
    let units: MLComputeUnits = switch arg("-units") {
    case "gpu": .cpuAndGPU
    case "all": .all
    case "cpu": .cpuOnly
    default: .cpuAndNeuralEngine
    }
    let tries = max(1, Int(arg("-retries") ?? "1") ?? 1)
    for fn in fns {
      for attempt in 1...tries {
        let c = MLModelConfiguration()
        c.computeUnits = units
        if !args.contains("-noFunctionName") { c.functionName = fn }
        var load = Load(fn: fn, attempt: attempt)
        let t = Date()
        do {
          var m: MLModel? = arg("-api") == "sync" ? try MLModel(contentsOf: url, configuration: c)
                                                  : try await MLModel.load(contentsOf: url, configuration: c)
          load.ok = m != nil
          m = nil
        } catch { load.error = "\(error)" }
        load.ms = Int(Date().timeIntervalSince(t) * 1000)
        (load.footprintMB, load.peakMB) = footprint()
        load.thermal = ProcessInfo.processInfo.thermalState.rawValue
        load.freeDiskMB = freeDiskMB()
        r.loads.append(load)
        save()
        if load.ok { break }
      }
    }
    let failed = fns.filter { fn in !r.loads.contains { $0.fn == fn && $0.ok } }
    r.status = failed.isEmpty ? "done" : "error"
  } catch {
    r.status = "error"
    r.loads.append(Load(fn: "setup", error: "\(error)", freeDiskMB: freeDiskMB()))
  }
  r.thermalEnd = ProcessInfo.processInfo.thermalState.rawValue
  r.freeDiskMBEnd = freeDiskMB()
  r.finishedAt = Date().ISO8601Format()
  save()
  try? "\(Date().timeIntervalSince1970)".write(to: stamp, atomically: true, encoding: .utf8)
  return "DONE \(r.status)"
}
