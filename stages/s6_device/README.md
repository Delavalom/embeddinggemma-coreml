# s6_device

PhotoSearchDemo's benchmark on a real iPhone: speed per image at the phone's own budget, speed per query, peak memory and
the thermal state after 1,000 images.

Run: `just stage 6` with an iPhone connected (USB or the same network), unlocked, trusted, with Developer Mode on.
Needs fresh passing s4 and s5. To choose a device:
`uv run --exact --group convert python -m stages.s6_device.run --device <UDID>`.

## Gate

- `min_devices` (≥ 1): devices that wrote a result in this run.
- `max_peak_rss_mb` (≤ 800): the maximum over devices of the app's peak `phys_footprint` (what iOS counts against
  the memory limit), sampled every 100 ms for the whole benchmark. The benchmark runs after the compile run, in a
  new process, so every function load is warm: the memory of every launch after the first (ADR-0009).
- `max_cold_compile_peak_mb` (≤ 2,100): the same measurement over the compile run, which loads every function once,
  one at a time, right after the install. It happens once per install or app update (ADR-0009).
- Recorded, not gated: `ms_per_image_70_median`, `ms_per_image_280_median`, `ms_per_query_median` (the worst
  device's medians), and in `details` the target `[s6_device.targets].target_ms_per_image_70` with
  `meets_target_70` per device, and `available_memory_mb_at_launch` per device (`os_proc_available_memory()`, the
  app's headroom before iOS ends it). Each device's full result (p90, first/cold value, model load time, thermal state
  after each budget, photos skipped) is in `details.devices` and `artifacts/s6_device/devices/<model>.json`.

## How the app gets the model

The model is not on Hugging Face yet (that is s7), so this stage bundles it:

1. `egcm.published.assemble` writes one published layout per image budget (ADR-0010) to
   `Examples/PhotoSearchDemo/Model/b70/` and `Model/b280/` (gitignored; `Model/` is replaced first): each budget's
   `EmbeddingGemma2.mlpackage/` and `tokenizer/` from its s4 device archive (sha256-checked, unpacked into
   `artifacts/s6_device/unpacked/b<budget>/`), and its own `files.json`. Recorded: `details.s4_recipe`,
   `details.package` and `details.files_json_sha256` per budget. `--package <path>` bundles one 14-function package
   flat in `Model/` (with the local tokenizer) for both budgets and records the recipe as null.
2. A build script in `project.yml` copies `Model/` into the app bundle when it exists.
3. The app loads each budget with `EmbeddingGemma.load(.file(bundleURL))`, which uses `Model/b<budget>/` (or `Model/`
   itself for a flat layout), verifies every file against that `files.json`, copies it to Application Support and
   compiles it once. Both launches report `model_revisions` = `{"70": ..., "280": ...}`, each
   `file-<first 16 hex of sha256(b<budget>/files.json)>` (`details.expected_revisions`), and the stage fails if
   either launch reports another model for either budget.

Without `Model/` the app falls back to the published default (`EmbeddingGemma.defaultRevision`), which fails until
s7 has run. The bundled app holds both budget packages uncompressed (the s4 report's
`details.budgets.<budget>.package_mb`), so it is about twice the size of one; a phone that downloads the model gets
only its budget's.

## What it does

1. Bundles the model (above) and writes `Examples/PhotoSearchDemo/Team.xcconfig` if missing
   (`python -m stages.s6_device.team`: the team id is the OU of the signing certificates listed by `asc`).
2. Picks the first `xcrun devicectl list devices` entry that is physical, iOS, with Developer Mode enabled.
   None: fails with "connect an iPhone with Developer Mode".
3. `xcodegen generate`, then `xcodebuild -configuration Release -destination id=<udid> -allowProvisioningUpdates
   build` (Release, because the timed path includes the Swift preprocessing). The old `.app` is deleted first and
   the new one checked with `codesign --verify`.
4. `devicectl device install app` (installing a new build discards the on-device compiles), then two launches,
   each `devicectl device process launch --console --terminate-existing <bundle> -- <mode> -runID <uuid>` (console
   in `artifacts/s6_device/console_<model><mode>.log`):
   - `-compile` (`CompileRunner.swift`): installs the phone's budget's model and loads every function of its package, one
     at a time, releasing each (the text functions twice, once per package). Writes `Documents/s6_compile.json`: peak, load time per function, thermal state, and
     `os_proc_available_memory()` at launch.
   - If the compile run fails, it is relaunched after a 15-second pause, up to 5 launches (`COMPILE_LAUNCHES`):
     functions it already compiled stay cached. On iOS 27.0.1 a process that fails a Neural Engine compile keeps
     failing it; a new process gets past it. While the phone is locked, a launch is retried every 30 s for up to an
     hour. `max_cold_compile_peak_mb` is the highest peak over the launches; the report keeps
     `launches` and `launch_errors`.
   - `-benchmark` (`BenchmarkRunner.swift`, below). Writes `Documents/s6_device.json`.
5. For each launch, polls `devicectl device copy from --domain-type appDataContainer --source <file>` every 15 s
   for up to 30 min. Only a file with this run's `run_id` counts; `status: error` fails at once with the app's
   message; the app exiting without a result fails too. The stage fails if either run used another model, or if
   a benchmark function load took over 30 s (a compile the compile run did not cache).

The benchmark (`BenchmarkRunner.swift`) runs the phone's own budget only (`ImageBudget.auto`: 280 on A17 Pro and
newer, 70 otherwise), the one package that phone ships with: the newest 1,000 photos (960 px from `PHImageManager`)
at budget 70, or the newest 200 at budget 280, then the 20 queries against them. Both packages are bundled, but
compiling all 18 functions on one iPhone 13 Pro Max (5.4 GB of compiled cache, iOS 27.0.1) made iOS evict the first
ones within minutes, so the benchmark's loads were compiles again. The other budget is measured on a phone of its
tier. Times cover `embed(image:)` only (preprocessing and
prediction, not the Photos fetch), after `prewarm()` (recorded as `load_ms`). The device name is not recorded.

After a run, keep the result: `python -m scripts.save_result s6_device` (writes `docs/results/s6_<model>.json` with
repo-relative paths and no home folder) and commit it.

## Known failure modes

- **"the benchmark compiled <function> ... after the compile run"**: iOS discarded a compile between the two
  launches (low storage, or an OS update in between). Rerun.
- **`max_cold_compile_peak_mb` fails, `max_peak_rss_mb` passes**: the package is too large to compile within the
  limit on this phone even though it runs fine afterwards. See ADR-0009.
- **"connect an iPhone with Developer Mode"**: not connected, locked, not trusted, or Developer Mode off.
- **Waits, then times out**: the Photos prompt was not answered. Tap **Allow Full Access** on the phone (first run
  only). Keep the phone unlocked: the app disables the idle timer, but a manual lock suspends it.
- **"Photos access was denied"**: allow it in Settings > Privacy > Photos > Photo Search, rerun.
- **xcodebuild signing errors**: delete `Team.xcconfig` and rerun (rewritten from `asc`), or open the project in
  Xcode once so it registers the device. `asc auth status --validate` must pass.
- **"the benchmark ran models {...}, expected {...}"** (or "the compile ran"): the app loaded an older install for
  that budget, or an app build that does not report `model_revisions` yet (None). Delete the app from the phone and
  rerun.
- **Few images measured**: the library has fewer than 1,000 photos, or iCloud photos could not be downloaded
  (`skipped` per budget).
