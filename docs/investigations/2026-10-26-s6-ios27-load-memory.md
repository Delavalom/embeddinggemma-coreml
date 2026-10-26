# s6 on iOS 27: flaky Neural Engine compiles, and ~1 GB per function load (2026-10-26)

Device: iPhone 13 Pro Max (A15, iPhone14,3), iOS 27.0.1. Mac: macOS 26.6.2, Xcode 27.1. Model: s4 `int8_table4`
(`files.json` 8a7c99c73b81ba34), one multifunction package, 14 functions, `model.mil` 51.6 MB, `weight.bin` 400 MB.
Probes: `scripts/probes/` (iOS app, Mac CLI, function extractor). Memory is `phys_footprint`, what iOS counts.

## 1. Flaky Neural Engine compiles: fixed (98d8610)

s6 failed in `prewarm` with `modelLoadFailed`: "`MLModelConfiguration`'s `.functionName` property must be `nil`
unless the model type is ML Program". Ruled out: bundling (the app bundle is complete), the on-device compile (its
`coremldata.bin` equals the Mac's up to metadata key order; `model.mil` and all 14 functions present), `.cpuOnly`
(always loads). The same `.cpuAndNeuralEngine` load failed in about 5 of 30 cold compiles, always 40-55 s in, with 0 to 5
other functions loaded, and succeeded on the next attempt every time. `FunctionLoad.withRetries` gives Neural Engine
loads 3 tries. With it the s6 benchmark completed budget 70 (1,000 photos, median 62 ms/image, p90 75 ms, queries
median 6.5 ms) and then failed budget 280 three times in a row, with the phone at thermal "serious" and the footprint
at 1.97 GB. `embed_image_280_square` alone, cool phone: OK first try (185 s, 125 MB).

## 2. Memory: every load of a 14-function model costs ~1 GB (open)

| Load (Mac, `.cpuAndNeuralEngine`) | 14-function package | 1-function package (`embed_text_16`) |
|---|---|---|
| cold (first compile) | 45-195 s, peak 0.85-2.0 GB | 45 s, peak 853 MB |
| warm (compile cached) | 6.5-7 s, peak **969 MB**, any function | **0.24 s, peak 45 MB** |
| `.cpuOnly` | 34 s, peak 2.0 GB | - |

- The warm peak does not depend on the function (text 16 and image 70 square both 969 MB), so Core ML processes the
  whole model on every load. Live heap after a load is 452 KB; the rest is freed `MALLOC_SMALL`/`MALLOC_LARGE`
  regions the allocator keeps (`malloc_zone_pressure_relief` returns 0 on macOS 26). The phone shows the same:
  ~120 MB after the first load, 1.3-1.5 GB after the second, 958 MB on a warm second load, peak 1.97 GB in s6.
- Cold compiles also need ~0.85-2 GB, once per function per install (an app reinstall/update recompiles: text
  ~60 s, image 70 ~130-240 s, image 280 ~185 s each on the phone).
- Gate: `[s6_device] max_peak_rss_mb = 800`. Not changed; any change needs an ADR the founder approves.

## 3. The warm cost scales with `model.mil` size (Mac, all functions of each package loaded in turn)

Sub-packages built with `scripts/probes/extract_functions.py`, measured with `mac_load_probe.swift`.

| Package | `model.mil` | weights | warm load per function | warm peak | cold peak |
|---|---|---|---|---|---|
| 1 text function | 1.6 MB | 213 MB | 0.24 s | 45 MB | 853 MB |
| 4 text functions | 6.5 MB | 214 MB | 0.9 s | 127 MB | 1,083 MB |
| 5 image-70 functions | 22.5 MB | 297 MB | 2.9 s | 424-541 MB | 973 MB |
| all 14 (shipped) | 51.6 MB | 400 MB | 6.5-7 s | 969 MB | up to 2.0 GB |

So warm loads cost about 20 MB of transient memory per MB of `model.mil`. Cold compiles cost about 0.85-1.1 GB whatever
the package, once per function per install. The s6 benchmark reinstalls the app, so its peak includes cold compiles.
That means a peak gate of 800 MB fails on a fresh install with any packaging. Gating the warm (steady-state) peak and
recording the cold-compile peak would be a gate change, so it needs an ADR.

## Next (not done)

1. ADR: how s6 measures memory (warm peak gated, cold compile recorded?), for the founder.
2. Pick the packaging: per-tier packages (text + one image budget, ~29 MB mil, one download of about today's size)
   (~29 MB mil, warm peak about 600 MB estimated) vs. separate text and image packages (cheapest loads, ~460 MB download, over the 360 MB gate) vs. fewer functions
   (e.g. enumerated shapes instead of one function per bucket). Then rerun s4-s6 (s4 = new packages; the download gate
   still applies).
3. Roll: first launch compiles for minutes; compile in the background, text functions first.
