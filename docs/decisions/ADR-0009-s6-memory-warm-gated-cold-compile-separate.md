# ADR-0009: s6 gates the steady-state memory peak, and measures the first compile separately

**Status:** proposed, 2026-10-26. Needs the founder's approval before `gates.toml [s6_device]` or
`stages/s6_device/` change.

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
     Threshold: 1,500 MB. Reasoning: the per-tier package is expected near 1.0-1.1 GB cold (the Mac's 4-function
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
- `gates.toml [s6_device]`: comment `max_peak_rss_mb` as the warm peak, add `max_cold_compile_peak_mb = 1500`.
- README: both measurements and the new failure modes.
- One commit, with this ADR marked accepted.

## Order of work

This ADR only decides what s6 measures. Next, independently: repackage the model (per-tier packages, text + one
image budget, about today's download, warm peak estimated ~600 MB from the Mac numbers), rerun s4 (the 360 MB
download gate still applies), s5, then s6 under whichever gate is accepted.

## Not established

- The jetsam limit for a foreground app on this iPhone or on 4 GB iPhones was not measured. The 1,500 MB ceiling
  is a judgment, not a measured limit. `os_proc_available_memory()` in the compile report will show the real
  headroom per device.
- Whether a `BGProcessingTask` gets enough memory to compile an image function in the background (Roll's plan).
  To be tested in Roll, not in s6.
- Whether iOS releases the compile's retained memory within the same process. Two launches avoid the question for
  the gate; Roll's single long-lived process does not.
