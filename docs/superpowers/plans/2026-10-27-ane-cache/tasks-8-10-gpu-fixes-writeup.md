<!-- Part of docs/superpowers/plans/2026-10-27-ios27-ane-cache-investigation.md; read that first. -->

## Task 8: GPU fallback measured on the phone (E3, U5)

**Files:**
- Modify: `Examples/PhotoSearchDemo/Sources/App.swift` (`-units` argument)
- Modify: `Examples/PhotoSearchDemo/Sources/CompileRunner.swift`, `BenchmarkRunner.swift` (record units)
- Create: `Examples/PhotoSearchDemo/Sources/AccuracyRunner.swift`
- Create: `scripts/probes/gpu_accuracy.py`

- [ ] **Step 1: Add the `-units` launch argument.** In `App.swift`, `DemoModel`:

```swift
import CoreML

  /// `-units gpu|cpu|all|ne` (default ne): compute units for every function this launch loads (E3, the GPU fallback).
  static let computeUnits: MLComputeUnits = {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: "-units"), i + 1 < a.count else { return .cpuAndNeuralEngine }
    switch a[i + 1] { case "gpu": return .cpuAndGPU; case "cpu": return .cpuOnly; case "all": return .all
                      default: return .cpuAndNeuralEngine }
  }()

  static func load(budget: EmbeddingGemma.ImageBudget = .auto) async throws -> EmbeddingGemma {
    try await EmbeddingGemma.load(source, budget: budget, dimensions: dimensions, computeUnits: computeUnits)
  }
```

  Add `var computeUnits = "\(DemoModel.computeUnits.rawValue)"` with the coding key `compute_units` to both
  `CompileResult` and `BenchmarkResult`. Build to confirm that `stages/s6_device` still decodes them (it ignores unknown
  keys; check `stages/s6_device/run.py`).
- [ ] **Step 2: Measure speed and memory, 3 repetitions.** Reinstall the demo (b70 bundled). For each repetition, run
  `-compile -units gpu -runID e3-cN` (load time of every function on its first GPU load), then
  `-benchmark -units gpu -runID e3-bN` (1,000 photos: ms/image median/p90, peak memory, thermal after). Then run the
  same `-compile`/`-benchmark` pair a second time **without** reinstalling (second-launch load time on the GPU: does the
  GPU path have a cache problem too?). Use the launch and copy commands from the handoff's "Tools" section. Record
  free disk before and after. Compare with the ANE numbers in `ev/compile-run11-9fn.json` and with ADR-0006 (Mac: GPU
  133 ms vs ANE 48 ms).
- [ ] **Step 3: Measure accuracy on the phone's GPU.** Add `-accuracy` to `App.swift`'s switch (next to `-compile`) to show
  `AccuracyView`:

```swift
// Examples/PhotoSearchDemo/Sources/AccuracyRunner.swift
import EmbeddingGemma
import ImageIO
import SwiftUI
import UIKit

/// `-accuracy [-units gpu]`: embeds every PNG in Documents/pixels (the s5 parity pixels, copied in with devicectl) at
/// 768 dimensions and writes Documents/accuracy.json {name: [Float]}, for scripts/probes/gpu_accuracy.py.
struct AccuracyView: View {
  @State private var status = "accuracy"
  var body: some View {
    Text(status).task {
      UIApplication.shared.isIdleTimerDisabled = true
      do {
        let model = try await EmbeddingGemma.load(DemoModel.source, budget: .tokens70, dimensions: 768,
                                                  computeUnits: DemoModel.computeUnits)
        let dir = URL.documentsDirectory.appending(path: "pixels")
        var out: [String: [Float]] = [:]
        for f in try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
          where f.pathExtension == "png" {
          guard let src = CGImageSourceCreateWithURL(f as CFURL, nil),
                let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
          out[f.deletingPathExtension().lastPathComponent] = try await model.embed(image: img)
          status = "\(out.count) images"
        }
        try JSONEncoder().encode(out).write(to: URL.documentsDirectory.appending(path: "accuracy.json"))
        status = "DONE \(out.count)"
      } catch { status = "FAILED \(error)" }
    }
  }
}
```

  Copy the 200 `*_70.png` files:
  `xcrun devicectl device copy to --device $U --domain-type appDataContainer --domain-identifier com.delavalom.egcm.demo --source /tmp/pixels70 --destination Documents/pixels`
  (first gather them with `mkdir /tmp/pixels70 && cp artifacts/s5_swift/pixels/*_70.png /tmp/pixels70/`). Run
  `-accuracy -units gpu`, then `-accuracy -units ne` on the same phone, so the comparison is on the phone's own
  numbers.

- [ ] **Step 4: Write the comparison script.**

```python
# scripts/probes/gpu_accuracy.py
"""usage: gpu_accuracy.py phone_gpu.json phone_ne.json   → cosine of each phone vector vs the s5 references."""
import json
import sys

import numpy as np

gpu, ne = (json.load(open(p)) for p in sys.argv[1:3])
runtime = np.load("artifacts/s5_swift/runtime.npz")   # Mac Swift ANE, keys image_iXXX_70
fp = np.load("artifacts/s5_swift/fp_70.npz")          # FP reference, keys iXXX (subset)


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


rows = {"gpu_vs_fp": [], "ne_vs_fp": [], "gpu_vs_mac_ne": [], "ne_vs_mac_ne": [], "gpu_vs_phone_ne": []}
for name, v in gpu.items():             # name like i012_70
    i = name.split("_")[0]
    if i in fp: rows["gpu_vs_fp"].append((cos(v, fp[i]), name))
    if i in fp and name in ne: rows["ne_vs_fp"].append((cos(ne[name], fp[i]), name))
    if f"image_{name}" in runtime:
        rows["gpu_vs_mac_ne"].append((cos(v, runtime[f"image_{name}"]), name))
        if name in ne: rows["ne_vs_mac_ne"].append((cos(ne[name], runtime[f"image_{name}"]), name))
    if name in ne: rows["gpu_vs_phone_ne"].append((cos(v, ne[name]), name))
for k, r in rows.items():
    if r:
        c = np.array([x for x, _ in r])
        print(f"{k:16} n={len(r):3} min={c.min():.5f} ({min(r)[1]}) p1={np.percentile(c, 1):.5f} mean={c.mean():.6f}")
```

  Run: `uv run --exact --group convert python -m scripts.probes.gpu_accuracy ~/egcm-evidence-raw/<date>/e3-acc-gpu.json ~/egcm-evidence-raw/<date>/e3-acc-ne.json | tee docs/investigations/evidence/2026-10-2x-ane-cache/e3-gpu-accuracy.txt`.
  Key formats were checked on 2026-10-27: `fp_70.npz` uses `iXXX` (170 of the 200 images), and `runtime.npz` uses
  `image_iXXX_<budget>`. Report the min for the hard images (i161 and the two from s3) by name.
- [ ] **Step 5: Write `e3-gpu.md`.** A table of 3 repetitions × (first-launch load per function, second-launch load,
  ms/image median/p90, peak MB, thermal after, free disk delta) for GPU and for ANE (from the same-day runs or
  `compile-run11`), plus the accuracy lines. Commit it with the demo changes:

```bash
git add Examples/PhotoSearchDemo/Sources scripts/probes/gpu_accuracy.py docs/investigations/evidence/2026-10-2x-ane-cache/e3-*
git commit -m "feat(demo): -units and -accuracy launch modes; evidence: GPU fallback measured on the phone"
```

## Task 9: Candidate fixes, each held to the same bar

Pick only candidates that the evidence from Tasks 5–8 points to. Each candidate gets a one-paragraph rationale that
cites the evidence before any code is written.

| ID | Candidate | Makes sense if | How it is tested |
|---|---|---|---|
| F-a | Keep the key's input stable (path, mtime, config: whatever E0b/E1 showed) | H3/H4/H5 confirmed | E1a matrix with the change, then the soak |
| F-b | Load 1-function packages without `functionName` (repackaging; measure download size) | E1c survives where b70 doesn't | E1d with 9 single-function packages; download size ≤ 360 MB |
| F-c | Before loading, detect skeleton entries in our own `Library/Caches/<id>/com.apple.e5rt.e5bundlecache` and delete them, then load (stops the `functionName` error, but still recompiles) | H6 confirmed by E2b | E2b with the fix: no error in 3/3 |
| F-d | `.cpuAndGPU` for image functions, or for all functions | E3 numbers are acceptable to the founder | E3 numbers + soak |
| F-e | Free-space gate + background compile in Roll | H2 confirmed (purge under pressure) | E1e with the gate |
| F-f | Whatever R1/R2 found (for example, a documented specialization or caching API) | a source says so | the same matrix |
| F-g | Apple bug: minimal repro app + Feedback Assistant draft (the founder files it) | the cache is lost with every input held constant and ≥ 8 GB free | the repro reproduces it 3/3 |

- [ ] **Step 1: Set the warm threshold from data.** Use the largest warm load seen in E0a/E1 (the Mac is 3.6 s,
  handoff F5 is about 6 s) × 1.5, rounded up, capped at 10 s. Write it in `e1-phone.md` before the soak starts.
- [ ] **Step 2: Soak test of the chosen fix.** Install the build with the fix. Run `phone_run.sh soak-NN -fns <all 9>`
  10 times over ≥ 24 h, with gaps of 10 min, 1 h, 3 h, overnight, and one reboot. **Pass:** every function is under the
  warm threshold on launches 2–10, no `functionName` error, and the s6 memory gates hold (warm peak ≤ 800 MB, ADR-0009).
  Use `phone_watch.sh` between launches to log the payload.
- [ ] **Step 3: If none passes,** write down which bar each candidate missed, by how much, and with which evidence.
  That is a result in its own right. The recommendation then chooses between F-d, F-g, and a "compile once per launch
  in the background" design, using the numbers.

## Task 10: Write-up, recommendation, certified s6

**Files:**
- Create: `docs/investigations/2026-10-2x-ane-cache.md`
- Create (only if packaging, gates, or compute units change): `docs/decisions/ADR-0011-<topic>.md` (status *proposed*)
- Modify: `ROLL-HANDOFF.md` (parent folder), `STATUS.md`

- [ ] **Step 1: Write the investigation report** with these sections: question; facts F1–F12 + N1–N5 (corrected
  where the experiments changed them); research table; experiments (every run, linked to its evidence file); answers
  to U1–U6, each labelled *answered (evidence)* or *open (what's missing)*; the GPU numbers; the recommendation and
  what it costs Roll (first-launch time, disk, memory, accuracy).
- [ ] **Step 2: If Apple's bug is the cause,** write `docs/investigations/feedback-draft-ios27-e5-cache.md` with the
  repro steps and the minimal app. **Ask the founder before filing anything.**
- [ ] **Step 3: Run the certified `just stage 6` once,** with the fix in place, only after the soak passes.
- [ ] **Step 4: Update `ROLL-HANDOFF.md` and memory** (`roll-and-embeddinggemma-coreml.md`) with the result. Commit and
  push `feat/rewrite` (the repo is private; pushing it is routine).

---

