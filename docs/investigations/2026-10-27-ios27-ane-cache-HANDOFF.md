# Handoff: iOS 27 Neural Engine compile cache — find the exact cause, with evidence (2026-10-27)

> **Plan written 2026-10-27 12:30:** `docs/superpowers/plans/2026-10-27-ios27-ane-cache-investigation.md` (start
> there). It corrects F2: a JSON listing shows that **no** finished cache entry on the phone holds compiled files.
> The 9 folders hold empty function folders (plan §0, N1; `ev/phone-e5-files-1215.txt`).

Read this whole file before doing anything. Then read `ROLL-HANDOFF.md` (repo parent folder) and
`docs/investigations/2026-10-26-s6-ios27-load-memory.md`.

## The founder's brief (verbatim — this is the mentality for the whole session)

> Okay I want you to create a handoff of these next two items. I want, for the handoff session that we're going to
> open, to create an intensive plan and do intensive research. It comes with evidence to understand what the exact
> issue is and what the exact number is. I don't want to work with a best guess. I want to work with evidence and I
> want to work with certainty.
>
> I want the next agent to create a plan, with details on research and evidence, on how to attack and approach the
> kind of issues that we're having. What's the final end solution that we can have to make this possible? Does that
> make sense? I want the next session to have the same mentality as this text.

The two items:
1. **Why compiled Neural Engine functions don't survive to the next app launch on the iPhone** (and why loads
   intermittently fail with the misleading `functionName` error). Exact cause, exact numbers.
2. **The fallback measured on the phone**: the same model on `.cpuAndGPU` (no Neural Engine compile): load time,
   ms/image, memory, accuracy, heat — so the founder can choose with numbers.

The end state: a written plan, research with sources, experiments with recorded evidence, and a recommended **final
solution** that makes the second and every later launch of Roll fast and reliable on the phone.

## Rules of evidence for this session

- Every claim in the write-up cites a measurement and the file it is in (`docs/investigations/evidence/...`). No
  "probably", "likely", "best guess" in conclusions; open questions are listed as open.
- One variable per experiment. Repeat each phone measurement at least 3 times; report every run, not the best.
- Record with every phone run: iOS build, thermal state, free disk (`volumeAvailableCapacityForImportantUsage`),
  `os_proc_available_memory()`, time since the previous launch, and the E5 cache listing before and after.
- Do not use the full `just stage 6` as the experiment loop (45–90 min per try). Use the small probes below; run the
  certified s6 once at the end.
- Standing rules still apply: never change `gates.toml` without an ADR the founder approves; ask before anything
  public; 16 GB Mac, one heavy process at a time; commit evidence as you go (global pre-commit blocks staged files
  over ~5k tokens: trim logs, keep raw copies outside the repo).

## What is established (with evidence)

Device: iPhone 13 Pro Max (A15, iPhone14,3), iOS 27.0.1 (24A446), 128 GB. Mac: M1 Pro, macOS 26.6.2, Xcode 27.1.
Model: s4 `int8_table4`, split per budget (ADR-0010): `EmbeddingGemma2-b70` (9 functions: `embed_text_{16,32,64,128}`,
`embed_image_70_{tall,portrait,square,landscape,wide}`), 343 MB download, `model.mil` 27.6 MB, `weight.bin` 356 MB.
Evidence folder: `docs/investigations/evidence/2026-10-27-ane-cache/` (below: `ev/`).

| # | Fact | Numbers | Evidence |
|---|---|---|---|
| F1 | On the phone, a later launch recompiles functions that an earlier launch compiled | relaunch 10 min after an 18-function compile: every function cold, 45–170 s each; relaunch ~20 min after a clean 9-function compile: `embed_text_16/32/64` 43.0 / 46.4 / 52.5 s (warm would be ~6 s) | `ev/relaunch-after-18fn-compile-1046.json`, `ev/relaunch-after-9fn-compile-1153.json` |
| F2 | **The cache key changes between launches for the same function** | E5 cache on the phone: `embed_text_16` compiled under key `072165CC…` (11:07) and again under a new key `10F6E2EA…` (11:51); `embed_text_32` under `C631AF80…` (11:08) and `887120C0…` (11:52). 26 key folders after ~1 h of launches; only 9 hold a function | `ev/phone-app-library-listing-1154.compact.txt` |
| F3 | Where the cache lives on iOS | `Library/Caches/com.delavalom.egcm.demo/com.apple.e5rt.e5bundlecache/24A446/<key>/<hash>.bundle/H14.bundle/<function>/<function>_{ane,bnns}` (app container, `Caches` = purgeable by iOS) | same listing |
| F4 | The installed model path is stable across launches | `Library/Application Support/EmbeddingGemma/local/file-99a7d34084f9186b/model.mlmodelc` (reused: `Installer.run` returns `store.installed(...)`; no rewrite seen, mtimes 10:57–10:58) | same listing; `swift/Sources/EmbeddingGemma/Installer.swift` |
| F5 | Sometimes the cache *did* survive | night of 10-26/27, compile relaunches 2–4 (same install, minutes apart): already-compiled functions loaded in ~6 s | `ev/s6-cert6.uniq.log` + the launch-4 progress readings in the session log (`embed_text_16` 6 s etc.) |
| F6 | One clean launch compiles all 9 functions fine | 9/9, no failure, no relaunch; peak `phys_footprint` 1,508 MB; free disk 4,745 → 3,151 MB; thermal serious | `ev/compile-run11-9fn.json` |
| F7 | The benchmark launch right after it failed | 13 min of loads, then `modelLoadFailed` (“`.functionName` property must be `nil` unless the model type is ML Program”), 0 images done; Photos access was granted (1,000 photos available) | `ev/benchmark-run11-failed.json`, `ev/console-benchmark-run11.log` |
| F8 | The error is intermittent within a process and sticky once it starts | failed after 2 functions at thermal *fair*; after 7 and 12 functions at *serious*; 3 in-process retries with 15–30 s backoff + thermal wait all failed; a new process got past it | `ev/s6-cert*.log`, `swift/Sources/EmbeddingGemma/FunctionLoad.swift` |
| F9 | Disk used by compiles | 18 compiles: 9,188 → 3,766 MB free (≈ 5.4 GB); 9 compiles: 4,745 → 3,151 MB (≈ 1.6 GB). After uninstalling the app the phone showed only ~1.9 GB free (not explained) | `ev/compile-run7-18fn.json`, `ev/compile-run11-9fn.json` |
| F10 | Memory gates are not the problem | cold-compile peak 1,417–1,516 MB (gate 2,100); Mac warm peak per-budget package 699 MB (gate 800, not yet measured on the phone) | `ev/compile-run*.json`, `ev/mac-tier-package-load-probe-2026-10-26.txt` |
| F11 | `ANECompilerService` exceeds a CPU-resource limit during compiles | “90 s CPU over 140 s … exceeding limit of 50% over 180 s”, *Action taken: none* | `ev/ANECompilerService.cpu_resource-0815-header.txt` |
| F12 | On the Mac, warm loads across processes work | per-budget package: second process 3.6–3.8 s per function, 699 MB peak (once, b70's first function recompiled: 57 s) | `ev/mac-tier-package-load-probe-2026-10-26.txt` |

## What is NOT known (each needs an experiment, not an opinion)

- **U1. What goes into the E5 cache key, and which input changes between launches** (F2). Candidates to test one at a
  time: model URL / inode / mtime of `model.mlmodelc` or its files; `MLModelConfiguration` fields
  (`functionName`, `computeUnits`, `parameters`, `optimizationHints`, low-precision flags); the load API
  (`MLModel.load(contentsOf:configuration:)` async vs sync init, `MLModelAsset`); multifunction vs single-function
  package; per-process state (container path, launch UUID); iOS build/bug.
- **U2. Why F5 worked and F1 didn't.** List every difference between those launches (time gap, launch mode, install
  vs no install, thermal, free disk, which functions) and test them.
- **U3. What triggers the `functionName` error** (F7, F8): cache miss + compile under pressure? A corrupt or partial
  cache entry? Concurrency with another compile? Needs device logs at the moment of failure.
- **U4. Does iOS purge `Library/Caches` E5 entries** under storage pressure, and when? (F3, F9; the ~1.9 GB after
  uninstall is unexplained.)
- **U5. GPU fallback numbers on the phone** (none yet; Mac only: ADR-0006, GPU 133 ms vs ANE 48 ms per image, GPU
  accuracy 0.9997 on the 2 hard images).
- **U6. Is this specific to iOS 27.0.1?** No other device/OS has been tried.

## Tools you have (use these, not full s6)

- **Phone app with progress files**: `Examples/PhotoSearchDemo` launch modes `-compile` (loads every function one at a
  time, writes `Documents/s6_compile.json` after each: per-function ms, peak, thermal, free disk) and `-benchmark`.
  Launch: `xcrun devicectl device process launch --terminate-existing --device <UDID> com.delavalom.egcm.demo --
  -compile -runID <id>`; read: `xcrun devicectl device copy from --device <UDID> --domain-type appDataContainer
  --domain-identifier com.delavalom.egcm.demo --source Documents/s6_compile.json --destination x.json`.
- **E5 cache listing on the phone** (the key evidence for U1): `xcrun devicectl device info files --device <UDID>
  --domain-type appDataContainer --domain-identifier com.delavalom.egcm.demo --subdirectory Library`. Take it before
  and after every launch.
- **Device logs** (for U3, and possibly the cache-key inputs): `sudo log collect --device-udid <UDID> --last 20m
  --output x.logarchive`, then `log show x.logarchive --predicate 'process == "ANECompilerService" OR process ==
  "aned" OR subsystem CONTAINS "coreml" OR subsystem CONTAINS "e5rt" OR process == "PhotoSearchDemo"'`. Also
  crash/resource reports: `devicectl device info files --domain-type systemCrashLogs`.
- **Probes**: `scripts/probes/ios-load-probe/` (a minimal iOS loader app), `scripts/probes/mac_load_probe.swift`,
  `scripts/probes/extract_functions.py` (build 1-function or N-function packages), `scripts/s6_dev.py` (s6's exact
  measurement without a report).
- **Mac E5 cache**: `~/Library/Caches/<executable>/com.apple.e5rt.e5bundlecache/<build>/<key>/…` (see
  `egcm/e5cache.py`). The Mac reuses keys across processes (F12): a cheap control for what “working” looks like.

## Plan to write first, then execute (suggested order; refine it in the plan document)

1. **Research (before touching the phone, ~1–2 h)**: Apple docs and WWDC sessions on Core ML model loading/caching
   (`MLModel.load`, multifunction models / `functionName`, `MLModelAsset`, `MLComputePlan`, “specialization” and
   cached compiled models), iOS 26/27 Core ML release notes, Apple Developer Forums and coremltools GitHub issues for
   “e5bundlecache”, “functionName must be nil”, “recompiles every launch”, “ANECompilerService”. Record each source
   with its URL and what it says. Also read Apple's guidance on `Library/Caches` purging.
2. **Mac control (E0)**: same b70 package, two processes, identical config; list the Mac E5 keys before/after.
   Expect the same key reused. Then change one input at a time (touch `model.mlmodelc` mtime; copy it to a new path;
   different `computeUnits`; `functionName` vs a 1-function package without it) and record which ones produce a new
   key. This identifies the key's inputs cheaply.
3. **Phone, minimal (E1)**: with `ios-load-probe` (or the demo's `-compile`), load ONE function in launch A, exit,
   load it again in launch B (seconds later), then after 10 min, after 1 h, after a reboot. E5 listing before/after
   each. 3 repetitions. Then the same with a 1-function package (no `functionName`). Then 9 functions.
4. **Phone logs at a cache miss and at a `functionName` failure (E2)**: `log collect` right after; extract the
   relevant lines into `evidence/`.
5. **GPU fallback on the phone (E3)**: `.cpuAndGPU` for image functions (and text, separately): per-launch load time
   for launch 1 and 2, ms/image median/p90 over the 1,000-photo benchmark, peak memory, thermal after, and accuracy
   (re-run the s5 comparison on GPU outputs if needed). Mac-side accuracy is in ADR-0006.
6. **Candidate fixes, each verified the same way**: e.g. stabilise whatever key input changes; load without
   `functionName` (per-function packages, with the download cost measured); `.cpuAndGPU` or hybrid; keep a
   long-lived process; anything the research turns up. Pass criterion for a fix: **10 consecutive launches over
   ≥ 24 h with every function loading warm** (state the warm threshold from data, e.g. ≤ 10 s per function), no
   `functionName` failures, s6 memory gates passing, download ≤ 360 MB.
7. **Write-up**: `docs/investigations/2026-10-2x-ane-cache.md` with facts, sources, experiments, numbers, and a
   recommendation for the **final solution** + an ADR if any gate or packaging changes. If it is an Apple bug: a
   minimal reproduction app and a Feedback Assistant draft (the founder files it; ask before anything public).
8. Then rerun certified s4→s6 only if needed (s4/s5 are fresh at 4384858; s6 needs a pass on the phone).

## State at handoff

- Repo `embeddinggemma-coreml-rewrite`, branch `feat/rewrite`, last commit 4384858 (pushed, private). s4 and s5
  passed (per-budget packages; s4 split identical bitwise; s5 runtime cosine 0.999999999999999, min cosine 0.9902).
  s6 not passed. `scripts/s6_dev.py` exists. s6 now: compile run first (up to 5 launches, 15 s apart; waits while
  the phone is locked; a dropped devicectl console is not an app exit), then the benchmark on the phone's own budget.
- Phone: demo app installed (b70 + b280 bundled), ~1.6 GB free at 11:53, many stale E5 key folders. Before
  experiments: note the free space, then uninstall the demo (`devicectl device uninstall app ...`) and restart the
  phone if free space is low; ask the founder to free ≥ 8 GB.
- Phone handling: USB cable (not Wi-Fi); unlocked before each launch (no Auto-Lock “Never” on this phone; the app
  disables the idle timer only while it runs); the demo app must stay in the foreground; after an uninstall the
  benchmark asks for Photos (the founder taps Allow Full Access).
- Roll consequence to carry into the solution: the first-launch compile is minutes and GB of disk; if the cache does
  not persist, every launch pays it. Roll must also check free space before setup.
