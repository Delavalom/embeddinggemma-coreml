# ADR-0009: s6 gates the steady-state memory peak, and measures the first compile separately

**Status:** accepted, 2026-10-26, founder approved option 3 with the cold-compile limit at **2,100 MB**, not the
1,500 MB proposed below: the 1,500 MB estimate was wrong (see "Decision (accepted)" at the end). Proposed the same day.

## Context

s6 (`just stage 6`) gates `max_peak_rss_mb <= 800`: the app's peak `phys_footprint`, sampled every 100 ms over the
whole benchmark (`BenchmarkRunner.swift`). Every s6 run reinstalls the app, so that window always contains the
first on-device compile of every function as well as the benchmark itself. On iOS 27.0.1 (iPhone 13 Pro Max) the
run peaked at 1.97 GB and failed.

The investigation (`docs/investigations/2026-10-26-s6-ios27-load-memory.md`) separated two costs:

| | what it is | when it happens | measured peak |
|---|---|---|---|
| **cold compile** | Core ML compiles a function for the Neural Engine | once per function, after each install or app update | 853 MB (1-function package) to 2.0 GB (14 functions), Mac; up to 1.97 GB on the phone |
| **warm load** | loading a function whose compile is cached | every launch | 45 MB (1 text function) to 969 MB (all 14), Mac; ~958 MB on the phone |

The warm cost scales with the package's `model.mil` (about 20 MB transient per MB of `model.mil`), so repackaging
fixes it. The cold cost does not: even the smallest package measured, one text function with 1.6 MB of `model.mil`,
peaks at 853 MB while compiling. **No packaging can pass an 800 MB gate that includes the first compile.** The gate
as written measures something we cannot change, and mixes it with something we can.

## What a user sees

- The first launch after install or update: one compile per function, minutes in total on the phone (text ~60 s,
  each image function 130-240 s). Roll will do this in the background, text first (ROLL-HANDOFF item 4).
- Every launch after that: warm loads only. That is the memory people live with, and the number other apps on the
  phone compete with.

## Options

1. **Keep the gate as is.** s6 fails on every packaging. Not viable.
2. **Gate the warm peak at 800 MB; record the cold peak, no gate.** Measures what repackaging controls. A regression
   in the compile (a bigger package, a new function, a new iOS) would go unnoticed until a phone kills the app.
3. **(Recommended) Gate both, separately, with the existing 800 MB unchanged for the warm peak.**
   - `max_peak_rss_mb <= 800` keeps its threshold and means the **warm** peak: a second launch of the app, with
     every function already compiled, running the full benchmark (1,000 images at 70, 20 queries, 200 at 280).
   - New `max_cold_compile_peak_mb`: the peak while compiling every function, one at a time, on a fresh install.
     Threshold as proposed: 1,500 MB (accepted as 2,100 MB; see the decision). Reasoning: the per-tier package is expected near 1.0-1.1 GB cold (the Mac's 4-function
     and 5-function sub-packages: 1,083 and 973 MB), and 1.5 GB stays under what iOS lets a foreground app use on
     6 GB phones with margin. It catches the 14-function package (2.0 GB) and any regression of that size. It is not
     a guarantee for 4 GB phones; see "Not established".
   - Recorded, not gated: per-function compile time, the thermal state after compiling, and
     `os_proc_available_memory()` at launch (the app's own headroom on that device).

## Recommendation

Option 3. The steady-state threshold does not move, and the one cost repackaging cannot remove gets a ceiling of its
own instead of disappearing from the report. It also gives Roll a number to plan its background compile around.

## Implementation (once accepted)

- The app gets a `-compileOnly` launch: install the model, compile every function one at a time (the retry from
  98d8610 still applies), write `s6_compile.json` (peak, per-function times, thermal state, available memory), exit.
- `stages/s6_device/run.py`: install, launch `-compileOnly`, wait for its result, then launch `-benchmark` in a new
  process (so the compile's freed-but-retained allocator memory is not counted against the warm run). Check that
  every function in the benchmark loaded warm (load time recorded per function; a cold compile in the second launch
  fails the stage).
- `gates.toml [s6_device]`: comment `max_peak_rss_mb` as the warm peak, add `max_cold_compile_peak_mb` (2100 as accepted).
- README: both measurements and the new failure modes.
- One commit, with this ADR marked accepted.

## Order of work

This ADR only decides what s6 measures. Next, independently: repackage the model (per-tier packages, text + one
image budget, about today's download, warm peak estimated ~600 MB from the Mac numbers), rerun s4 (the 360 MB
download gate still applies), s5, then s6 under whichever gate is accepted.

## Not established

- The jetsam limit for a foreground app on this iPhone or on 4 GB iPhones was not measured. The cold ceiling (1,500 MB proposed, 2,100 MB accepted)
  is a judgment, not a measured limit. `os_proc_available_memory()` in the compile report will show the real
  headroom per device.
- Whether a `BGProcessingTask` gets enough memory to compile an image function in the background (Roll's plan).
  To be tested in Roll, not in s6.
- Whether iOS releases the compile's retained memory within the same process. Two launches avoid the question for
  the gate; Roll's single long-lived process does not.

## Decision (accepted, 2026-10-26)

The founder accepted option 3: two gates, measured in two launches. Before implementing it, the per-budget
packages planned under "Order of work" were measured on the Mac (M1 Pro, macOS 26.6.2, `.cpuAndNeuralEngine`,
`scripts/probes/extract_functions.py` + `mac_load_probe.swift`; each function loaded and released in turn):

| package (functions) | device download (Apple Archive, with the tokenizer) | warm load per function | warm peak | cold peak |
|---|---|---|---|---|
| all 14 (shipped by s4 today) | 358.7 MB | 6.5-7 s | 969 MB | ~2.0 GB |
| text + budget 70 (9) | 343.1 MB | 3.6-3.8 s | (see below) | 1,961 MB |
| text + budget 280 (9) | 352.8 MB | 3.6-3.8 s | **699 MB** | 2,019 MB |
| text only (4) + image 70 only (5), two packages | 198.3 + 277.3 MB | 0.9 s / 2.9 s | 127 MB / 424-541 MB | 1,083 MB / 973 MB |

- The 1,500 MB proposal assumed the per-budget package would compile at about 1.0-1.1 GB, like the 4- and
  5-function sub-packages. It does not: a package holding both text and image functions compiles at about 2 GB, the
  same as the 14-function one, from its very first function. Only separate text and image packages stay near 1 GB,
  and they download about 475 MB together (the shared decoder weights are duplicated), over s4's 360 MB.
- Budget 70's warm run recompiled its first function (57 s, 1.9 GB), so its warm peak was not measured; the other 8
  loads were warm (3.6-3.8 s). The disk had 12 GB free; whether Core ML evicted the compile because of that is not
  established. s6's `cold_loads` check fails a run in which this happens on the phone.

The founder chose per-budget packages with the cold limit at the measured reality plus a margin. It still fails a
regression the size of a bigger model, and the s6 report records each phone's `os_proc_available_memory()` next to
it. Their reasoning: the app's users (people with very large photo libraries) mostly have recent phones, and per-
budget packages help every phone (each downloads only its budget; warm peak ~970 to ~700 MB; loads 7 s to 3.6 s).

| key | threshold | what it measures |
|---|---|---|
| `max_peak_rss_mb` | `<= 800` (unchanged) | the benchmark's peak, in a launch after the compile run (every load warm) |
| `max_cold_compile_peak_mb` | `<= 2100` (new) | the compile run's peak: every function compiled once, one at a time, after the install |

Known gap: 4 GB iPhones (12, 13, 13 mini, SE 3) were not tested. iOS may end an app there before a 2 GB compile
finishes. The App Store cannot require an amount of RAM (its device capabilities are by chip), so excluding those
phones would be an in-app check. That is decided before Roll's launch, with a measurement from a 4 GB phone if one
is available. The fallback is separate text and image packages, at the download cost above.
