<!-- Part of docs/superpowers/plans/2026-10-27-ios27-ane-cache-investigation.md; read that first. -->

## Task 3: Phone probe v2 and its run script

The probe changes one thing per launch and records the environment the rules require.

**Files:**
- Modify (rewrite): `scripts/probes/ios-load-probe/Sources/Probe.swift`
- Modify: `scripts/probes/ios-load-probe/project.yml` (bundle the probe models)
- Create: `scripts/probes/ios-load-probe/build-models.sh`, `scripts/probes/phone_run.sh`, `scripts/probes/phone_watch.sh`

- [ ] **Step 1: Build the probe models.** b70 (9 functions) and a 1-function `embed_text_16` package (for H5,
  loaded without `functionName`).

```bash
# scripts/probes/ios-load-probe/build-models.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SRC=Examples/PhotoSearchDemo/Model/b70/EmbeddingGemma2.mlpackage
OUT=scripts/probes/ios-load-probe/ProbeModels; mkdir -p $OUT
rsync -a --delete $SRC/ $OUT/b70.mlpackage/
uv run --exact --group convert python scripts/probes/extract_functions.py $SRC $OUT/t16.mlpackage embed_text_16
du -sh $OUT/*
```

Add `ProbeModels/` to `.gitignore` (about 500 MB). In `project.yml`, under `targets.Probe.sources`, add
`- path: ProbeModels` with `type: folder` and `buildPhase: resources`, so the `.mlpackage` folders are copied as they
are and *not* compiled by Xcode. The probe compiles them on the device, the way `ModelStore` does.

- [ ] **Step 2: Rewrite `Probe.swift`.**

```swift
import CoreML
import SwiftUI
import UIKit
import os

@main struct ProbeApp: App {
  @State private var status = "probe"
  var body: some Scene {
    WindowGroup {
      Text(status).task {
        UIApplication.shared.isIdleTimerDisabled = true  // the probe must stay in front and awake
        status = await runProbe()
      }
    }
  }
}

// Launch arguments (one variable per launch):
//   -tag X              output Documents/probe-X.json
//   -model b70|t16      which bundled package (default b70)
//   -fns a,b            functions to load, one at a time, each released before the next
//   -noFunctionName     load without config.functionName (the 1-function package t16)
//   -units ne|gpu|cpu|all
//   -api async|sync     MLModel.load(contentsOf:configuration:) vs MLModel(contentsOf:configuration:)
//   -from as|bundle     load the compiled model from Application Support (default, like ModelStore) or from tmp
//   -retries N          tries per function (default 1: the experiments want the raw failure)
struct Load: Codable {
  var fn: String; var attempt: Int; var ok: Bool; var ms: Int; var error: String?
  var footprintMB: Int; var peakMB: Int; var thermal: Int; var freeDiskMB: Int
}
struct Report: Codable {
  var tag: String; var args: [String]; var osBuild: String; var startedAt: String; var finishedAt = ""
  var secondsSinceLastProbe: Double?; var thermalStart: Int; var thermalEnd = 0
  var freeDiskMBStart: Int; var freeDiskMBEnd = 0; var availMemMBStart: Int
  var modelPath = ""; var modelFiles: [String: [String: String]] = [:]   // file → size, mtime, inode
  var compileMS: Int?; var loads: [Load] = []; var status = "running"
}

func footprint() -> (Int, Int) {
  var info = task_vm_info_data_t()
  var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
  let kr = withUnsafeMutablePointer(to: &info) {
    $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
  }
  guard kr == KERN_SUCCESS else { return (-1, -1) }
  return (Int(info.phys_footprint / 1_048_576), Int(info.ledger_phys_footprint_peak / 1_048_576))
}
func freeDiskMB() -> Int {
  let v = try? URL.documentsDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
  return Int((v?.volumeAvailableCapacityForImportantUsage ?? -1) / 1_048_576)
}
func osBuild() -> String {
  var size = 0; sysctlbyname("kern.osversion", nil, &size, nil, 0)
  var b = [CChar](repeating: 0, count: size); sysctlbyname("kern.osversion", &b, &size, nil, 0)
  return String(cString: b)
}
func fileFacts(_ dir: URL) -> [String: [String: String]] {
  var out: [String: [String: String]] = [:]
  for name in ["model.mil", "weights/weight.bin", "coremldata.bin"] {
    let p = dir.appending(path: name).path
    guard let a = try? FileManager.default.attributesOfItem(atPath: p) else { continue }
    out[name] = ["size": "\(a[.size] ?? "")", "mtime": "\((a[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)",
                 "inode": "\(a[.systemFileNumber] ?? "")"]
  }
  return out
}

func runProbe() async -> String {
  let args = CommandLine.arguments
  func arg(_ k: String) -> String? { args.firstIndex(of: k).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
  let tag = arg("-tag") ?? "x", modelName = arg("-model") ?? "b70"
  let out = URL.documentsDirectory.appending(path: "probe-\(tag).json")
  let stamp = URL.documentsDirectory.appending(path: "last-probe-end.txt")
  let last = (try? String(contentsOf: stamp, encoding: .utf8)).flatMap(Double.init)
  var r = Report(tag: tag, args: Array(args.dropFirst()), osBuild: osBuild(), startedAt: Date().ISO8601Format(),
                 secondsSinceLastProbe: last.map { Date().timeIntervalSince1970 - $0 },
                 thermalStart: ProcessInfo.processInfo.thermalState.rawValue, freeDiskMBStart: freeDiskMB(),
                 availMemMBStart: Int(os_proc_available_memory() / 1_048_576))
  func save() { if let d = try? JSONEncoder().encode(r) { try? d.write(to: out, options: .atomic) } }
  save()
  do {
    // Compile once per package into Application Support/probe/<model>.mlmodelc (like ModelStore); reuse after.
    let asDir = URL.applicationSupportDirectory.appending(path: "probe")
    let compiled = asDir.appending(path: "\(modelName).mlmodelc")
    if !FileManager.default.fileExists(atPath: compiled.path) {
      let src = Bundle.main.url(forResource: "ProbeModels/\(modelName)", withExtension: "mlpackage")!
      let t = Date()
      let tmp = try await MLModel.compileModel(at: src)
      try FileManager.default.createDirectory(at: asDir, withIntermediateDirectories: true)
      try FileManager.default.moveItem(at: tmp, to: compiled)
      r.compileMS = Int(Date().timeIntervalSince(t) * 1000)
    }
    var url = compiled
    if arg("-from") == "bundle" {  // H4: same bytes, different path each launch
      url = FileManager.default.temporaryDirectory.appending(path: "\(UUID().uuidString).mlmodelc")
      try FileManager.default.copyItem(at: compiled, to: url)
    }
    r.modelPath = url.path; r.modelFiles = fileFacts(url); save()
    let units: MLComputeUnits = switch arg("-units") { case "gpu": .cpuAndGPU; case "all": .all; case "cpu": .cpuOnly; default: .cpuAndNeuralEngine }
    let tries = Int(arg("-retries") ?? "1") ?? 1
    for fn in (arg("-fns") ?? "").split(separator: ",").map(String.init) {
      for attempt in 1...max(1, tries) {
        let c = MLModelConfiguration(); c.computeUnits = units
        if !args.contains("-noFunctionName") { c.functionName = fn }
        let t = Date()
        var load = Load(fn: fn, attempt: attempt, ok: false, ms: 0, footprintMB: 0, peakMB: 0, thermal: 0, freeDiskMB: 0)
        do {
          var m: MLModel? = arg("-api") == "sync" ? try MLModel(contentsOf: url, configuration: c)
                                                  : try await MLModel.load(contentsOf: url, configuration: c)
          load.ok = true; _ = m; m = nil
        } catch { load.error = "\(error)" }
        load.ms = Int(Date().timeIntervalSince(t) * 1000)
        (load.footprintMB, load.peakMB) = footprint()
        load.thermal = ProcessInfo.processInfo.thermalState.rawValue; load.freeDiskMB = freeDiskMB()
        r.loads.append(load); save()
        if load.ok { break }
      }
    }
    r.status = r.loads.allSatisfy { $0.ok || $0.attempt < tries } ? "done" : "error"
  } catch { r.status = "error"; r.loads.append(Load(fn: "setup", attempt: 1, ok: false, ms: 0, error: "\(error)",
                                                     footprintMB: 0, peakMB: 0, thermal: 0, freeDiskMB: 0)) }
  r.thermalEnd = ProcessInfo.processInfo.thermalState.rawValue; r.freeDiskMBEnd = freeDiskMB()
  r.finishedAt = Date().ISO8601Format(); save()
  try? "\(Date().timeIntervalSince1970)".write(to: stamp, atomically: true, encoding: .utf8)
  return "DONE \(r.status)"
}
```

Remove the old `-hold`/`-seq` modes. `run.sh` stays in git history and is replaced by `phone_run.sh`.

- [ ] **Step 3: Write `phone_run.sh`.** It lists the cache, launches, waits, lists again, and saves everything.

```bash
#!/bin/zsh
# usage: phone_run.sh TAG <probe args...>   e.g. phone_run.sh e1a-r1-A -fns embed_text_16
# Writes ~/egcm-evidence-raw/<date>/<TAG>/{pre,post}.json, probe.json, summary.json; prints the summary line.
set -u
U=${UDID:-00008110-0014594C0A98401E}; APP=com.delavalom.egcm.demo; TAG=$1; shift
D=~/egcm-evidence-raw/$(date +%F)/$TAG; mkdir -p $D
REPO=$(git -C "$(dirname $0)" rev-parse --show-toplevel)
list() { xcrun devicectl device info files --device $U --domain-type appDataContainer --domain-identifier $APP \
           --subdirectory Library --json-output $1 >/dev/null 2>&1; }
list $D/pre.json
date -u +%FT%TZ > $D/launched-at.txt
xcrun devicectl device process launch --device $U --terminate-existing $APP -- -tag $TAG "$@" >/dev/null 2>&1 \
  || { echo "$TAG LAUNCH FAILED (phone locked?)"; exit 1; }
for i in {1..240}; do   # up to 60 min
  sleep 15
  xcrun devicectl device copy from --device $U --domain-type appDataContainer --domain-identifier $APP \
    --source Documents/probe-$TAG.json --destination $D/probe.json >/dev/null 2>&1 || continue
  grep -q '"status":"running"' $D/probe.json || break
done
list $D/post.json
cd $REPO && uv run --exact --group convert python - $D <<'EOF'
import json, sys
from pathlib import Path
from scripts.probes.e5_listing import diff, load
d = Path(sys.argv[1]); p = json.loads((d / "probe.json").read_text())
pre, post = load(str(d / "pre.json")), load(str(d / "post.json"))
s = {"tag": p["tag"], "status": p["status"], "os": p["osBuild"], "since_last_s": p.get("secondsSinceLastProbe"),
     "thermal": [p["thermalStart"], p["thermalEnd"]], "free_mb": [p["freeDiskMBStart"], p["freeDiskMBEnd"]],
     "avail_mem_mb": p["availMemMBStart"], "compile_ms": p.get("compileMS"),
     "loads": [(l["fn"], l["ok"], l["ms"], l["peakMB"], (l.get("error") or "")[:80]) for l in p["loads"]],
     "cache_pre": pre["totals"], "cache_post": post["totals"], "cache_diff": diff(pre, post)}
(d / "summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps({k: s[k] for k in ("tag", "status", "since_last_s", "free_mb", "loads", "cache_post")}))
EOF
```

- [ ] **Step 4: Write `phone_watch.sh`.** It lists the cache every N minutes with no launch (for H1/H2).

```bash
#!/bin/zsh
# usage: phone_watch.sh TAG MINUTES COUNT   lists the E5 cache COUNT times, MINUTES apart, without launching anything
U=${UDID:-00008110-0014594C0A98401E}; TAG=$1; D=~/egcm-evidence-raw/$(date +%F)/$TAG; mkdir -p $D
REPO=$(git -C "$(dirname $0)" rev-parse --show-toplevel)
for i in $(seq 1 $3); do
  f=$D/watch-$(date +%H%M%S).json
  xcrun devicectl device info files --device $U --domain-type appDataContainer --domain-identifier com.delavalom.egcm.demo \
    --subdirectory Library --json-output $f >/dev/null 2>&1
  (cd $REPO && uv run --exact --group convert python -c "import json,sys; from scripts.probes.e5_listing import load; print(sys.argv[1][-11:-5], json.dumps(load(sys.argv[1])['totals']))" $f)
  [ $i -lt $3 ] && sleep $(( $2 * 60 ))
done
```

- [ ] **Step 5: Build, install, and run a smoke test.**

```bash
chmod +x scripts/probes/phone_run.sh scripts/probes/phone_watch.sh scripts/probes/ios-load-probe/build-models.sh
scripts/probes/ios-load-probe/build-models.sh
cd scripts/probes/ios-load-probe && mise exec -- xcodegen generate && cd -
xcodebuild -project scripts/probes/ios-load-probe/Probe.xcodeproj -scheme Probe -destination id=00008110-0014594C0A98401E \
  -allowProvisioningUpdates -derivedDataPath /tmp/probe-dd build
xcrun devicectl device install app --device 00008110-0014594C0A98401E /tmp/probe-dd/Build/Products/Debug-iphoneos/PhotoSearchDemo.app
scripts/probes/phone_run.sh smoke -model b70 -fns embed_text_16 -units cpu
```

Expected: `status done`, one load `ok true`, `free_mb` and `since_last_s` (null on the first run) filled in. The `cpu`
run creates no ANE compile, so it mainly checks the tools. If `devicectl` prints a different `--json-output` schema,
fix `from_devicectl` and its test first.

- [ ] **Step 6: Commit.**

```bash
git add scripts/probes .gitignore
git commit -m "feat(probes): phone probe v2 (one variable per launch, environment record) and run/watch scripts"
```

