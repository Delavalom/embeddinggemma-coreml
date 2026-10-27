# iOS 27 Neural Engine cache investigation: plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.
> Read `docs/investigations/2026-10-27-ios27-ane-cache-HANDOFF.md` first. Its brief and rules of evidence apply to
> every step here.

**Goal:** find out with evidence why compiled Neural Engine functions don't survive to the next launch on the iPhone
(iOS 27.0.1), what triggers the `functionName` load error, and what the `.cpuAndGPU` fallback costs on the phone. Then
recommend a final solution that passes 10 warm launches in a row over 24 hours or more.

**Architecture:** experiments, not features. First a small set of tools: a cache-listing parser, a phone probe app
that loads one variable at a time and records its environment, run scripts that list the cache before and after each
launch, and a log extractor. Then experiments in order of cost: the Mac first (the control), then single-function
tests on the phone, then logs, then the GPU numbers, then candidate fixes. Every run writes a raw file outside the
repo and a short summary into `docs/investigations/evidence/2026-10-2x-ane-cache/`.

**Tech stack:** Swift 6 / Core ML (iOS 18+ app, macOS command-line tool), Python 3.12 via `uv`, `xcrun devicectl`,
`log collect` / `log show`, xcodegen.

---

## 0. New evidence found while writing this plan (2026-10-27, 12:15 AST)

Read-only checks done while writing the plan. Nothing was launched on the phone. These checks change where to start.

| # | Finding | Numbers | Evidence |
|---|---|---|---|
| N1 | **No finished cache entry on the phone holds a compiled payload.** The only E5 regular files are 13 `model.milhash` files plus the files of the one unfinished `.tmp.` bundle (`embed_text_128`, killed mid-compile). Every finished `<hash>.bundle/H14.bundle/<fn>/<fn>_{ane,bnns}` folder is empty (64 bytes on APFS). None has an `H14.e5` file, and no text function has a `bnns_program.bnnsir`. | 26 key folders, 95 E5 entries, 17 regular files; 0 finished bundles with a payload | `phone-e5-files-1215.txt` (from `devicectl … --json-output`; raw JSON at `~/egcm-evidence-raw/2026-10-27/listing-now.json`) |
| N2 | On the Mac, a finished bundle normally holds `H13S.bundle/H13S.e5` (5 KB) and, for text functions, `<fn>_bnns/bnns_program.bnnsir` (256 MB). `_ane` folders are empty on the Mac too, so the compiled ANE program isn't stored in this folder. It is probably in `aned`'s own system cache, which still needs confirming (R1, E0a). | 36 bundles, 268 MB each for text functions with payload | `mac-probe-e5-structure-1215.txt` |
| N3 | The Mac also has payload-less bundles and the same function under many keys (`embed_text_16` under F34CE88D, A0DBBFAC, 77155838, CC80C65F, E319681F on 10-26). The probes that day used several different packages, so this alone doesn't show key churn on the Mac. | 36 bundles, about 20 without payload | same file |
| N4 | `model.milhash` is a 64-hex string equal to its own folder's name on all 34 Mac folders. It is **not** `sha256(model.mil)` of any of the 40 `model.mil` files written on this Mac since 10-20. | 34/34 equal, 0/40 matches | `mac-probe-e5-structure-1215.txt`; this check is repeated properly in Task 4 |
| N5 | The phone made 13 `model.milhash` folders between 10:58 and 11:34 AST, but only one package (b70) was compiled that morning. So either the milhash changes from one load to the next, or something else is hashed. This is open. | 13 milhash folders, 1 package | `phone-e5-files-1215.txt` |

**What this changes:** handoff fact F2 says "only 9 [key folders] still held a compiled function". The JSON listing
shows that the 9 hold *function folders*, but none of them holds compiled files. So there are two different possible
failures, and the plan has to tell them apart before anything else:

- **(A) Payload removed:** the entry is written and later emptied (by iOS's cache purge, by E5 itself, or by the
  compile process on exit), so the next launch can't use it and compiles again under a new key.
- **(B) Key churn:** the key differs from one launch to the next even though the payload is still there.

Both may be true. The first phone experiment (Task 7) is designed to separate them.

## 1. Hypotheses and the experiment that decides each one

Each hypothesis is stated so that a measurement can prove it wrong. None of them is a conclusion.

| H | Hypothesis | Decided by | Prediction if true |
|---|---|---|---|
| H1 | The payload of a finished entry is removed after the compile, with no launch involved | E1a: list the cache every few minutes after a one-function launch, with the app not running and ≥ 8 GB free | `H14.e5`/`bnnsir` present right after, gone later |
| H2 | The payload is removed only under storage pressure (iOS purges `Library/Caches`) | E1a at ≥ 8 GB free vs E1e at < 2 GB free (same steps) | survives at 8 GB, removed at < 2 GB |
| H3 | The cache key includes an input that changes between launches even when the payload is kept (process, launch, time) | E0b on the Mac, then E1a launch B seconds after A with the payload still present | new key folder on launch B although the old entry is intact |
| H4 | The key includes file metadata of `model.mlmodelc` (path, inode, mtime) | E0b one at a time: touch, copy, rename | new key only on the metadata change |
| H5 | The key or the payload depends on the load API or `MLModelConfiguration` (`functionName`, units, sync vs async) | E0b + E1c (1-function package without `functionName`) | survival differs by API or config |
| H6 | The `functionName` error is what Core ML reports when it loads against an entry with no payload (or a partial one) | E2b: build that state on purpose, then load | the error appears exactly in that state |
| H7 | iOS 27.0.1 itself: the same steps keep the cache on another OS | Mac (macOS 26.6.2) vs phone. A second device or OS only if one is available (ask) | open unless another device is available |

## 2. Order, time box, and what the founder has to do

| Phase | Tasks | Time box | Phone? | Founder |
|---|---|---|---|---|
| Setup | 1 | 30 min | yes | free ≥ 8 GB on the phone; unlock it |
| Tools | 2, 3 | 2 h | no | — |
| Research | 4 (runs alongside Tools) | 2 h | no | — |
| Mac control | 5 | 2 h | no | run 2 `sudo` commands |
| Phone, minimal | 6, 7 | 4 h, then waits (1 h, a reboot) | yes | unlock; one reboot |
| Logs | 8 | 1.5 h | yes | `sudo log collect` |
| GPU | 9 | 3 h | yes | Photos "Allow Full Access" after the install |
| Fixes + soak | 10 | 1 h of work + ≥ 24 h of waiting | yes | unlock about 10 times over a day |
| Write-up | 11 | 2 h | no | read; approve any ADR |

Stop rule: if any phase runs over twice its time box, write down what is known, then ask the founder whether to
continue, re-scope, or switch to the fallback.

---

## Task files

The tasks are split into files so that each one stays under the repo's 5k-token commit limit. Do them in order.

| File | Tasks |
|---|---|
| [`2026-10-27-ane-cache/tasks-1-2-baseline-parser.md`](2026-10-27-ane-cache/tasks-1-2-baseline-parser.md) | 1 phone baseline and reset · 2 `e5_listing.py` parser (TDD) |
| [`2026-10-27-ane-cache/task-3-phone-probe.md`](2026-10-27-ane-cache/task-3-phone-probe.md) | 3 phone probe v2, `phone_run.sh`, `phone_watch.sh` |
| [`2026-10-27-ane-cache/tasks-4-7-research-mac-phone-logs.md`](2026-10-27-ane-cache/tasks-4-7-research-mac-phone-logs.md) | 4 research · 5 Mac control (E0) · 6 phone E1 matrix · 7 logs + `functionName` repro (E2) |
| [`2026-10-27-ane-cache/tasks-8-10-gpu-fixes-writeup.md`](2026-10-27-ane-cache/tasks-8-10-gpu-fixes-writeup.md) | 8 GPU on the phone (E3) · 9 candidate fixes + 24 h soak · 10 write-up, ADR, certified s6 |

## Self-check against the handoff

| Handoff item | Where |
|---|---|
| Item 1, exact cause (U1, U2, U4) | Tasks 5, 6, 7 + N1–N5 |
| `functionName` error (U3) | Task 7 (E2, E2b) |
| Item 2, GPU on the phone (U5): load, ms/image, memory, accuracy, heat | Task 8 |
| U6, iOS 27 specific | H7: the Mac comparison only; a second device only if available (ask the founder) |
| Research with sources | Task 4 |
| One variable per run, ≥ 3 repetitions, every run reported | Tasks 5–8 (r1–r3, every run in a table) |
| Thermal, free disk, memory, time since last launch, cache listing before and after | `Probe.swift` report + `phone_run.sh` |
| No full s6 as the experiment loop | only Task 10 Step 3, once |
| Fix bar: 10 launches over ≥ 24 h, all warm | Task 9 Step 2 |
| Commit evidence as you go; staged files < 5k tokens | every task commits trimmed files; raw files in `~/egcm-evidence-raw/` |
