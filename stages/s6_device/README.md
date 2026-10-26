# s6_device

PhotoSearchDemo's benchmark on a real iPhone: speed per image at both budgets, speed per query, peak memory and
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

1. `egcm.published.assemble` writes the published layout to `Examples/PhotoSearchDemo/Model/` (gitignored):
   `EmbeddingGemma2.mlpackage/` and `tokenizer/` from s4's device archive (sha256-checked, unpacked into
   `artifacts/s6_device/unpacked/`, recipe recorded as `details.s4_recipe`; `--package <path>` bundles another
   package with the local tokenizer and records null), and `files.json`.
2. A build script in `project.yml` copies `Model/` into the app bundle when it exists.
3. The app loads it with `EmbeddingGemma.load(.file(bundleURL))`, which verifies every file against `files.json`,
   copies it to Application Support and compiles it once. The app reports `model_revision` =
   `file-<first 16 hex of sha256(files.json)>`, and the stage fails if that is not the model it just bundled.

Without `Model/` the app falls back to the published default (`EmbeddingGemma.defaultRevision`), which fails until
s7 has run. The bundled app is about the size of the uncompressed package (the s4 report's
`details.variants.<recipe>.package_mb`; `int8_table4` zips to about 350 MB).

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
   - `-compile` (`CompileRunner.swift`): installs the model and loads every function the benchmark uses, one at a
     time, releasing each. Writes `Documents/s6_compile.json`: peak, load time per function, thermal state, and
     `os_proc_available_memory()` at launch.
   - `-benchmark` (`BenchmarkRunner.swift`, below). Writes `Documents/s6_device.json`.
5. For each launch, polls `devicectl device copy from --domain-type appDataContainer --source <file>` every 15 s
   for up to 30 min. Only a file with this run's `run_id` counts; `status: error` fails at once with the app's
   message; the app exiting without a result fails too. The stage fails if either run used another model, or if
   a benchmark function load took over 30 s (a compile the compile run did not cache).

The benchmark (`BenchmarkRunner.swift`): the newest 1,000 photos (960 px from `PHImageManager`) at budget 70, the
20 queries against them, then the newest 200 at budget 280. Times cover `embed(image:)` only (preprocessing and
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
- **"the app ran model file-..., expected file-..."**: the app loaded an older install. Delete the app from the
  phone and rerun.
- **Few images measured**: the library has fewer than 1,000 photos, or iCloud photos could not be downloaded
  (`skipped` per budget).
