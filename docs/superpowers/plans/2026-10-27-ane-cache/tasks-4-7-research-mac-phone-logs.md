<!-- Part of docs/superpowers/plans/2026-10-27-ios27-ane-cache-investigation.md; read that first. -->

## Task 4: Research with sources (runs alongside Tasks 2–3)

**Files:**
- Create: `docs/investigations/2026-10-2x-ane-cache-research.md`

Give R1 and R2 to two research subagents in parallel (the brief below is everything they get). Do R3 and R4 in the
main session. Each source gets a row: **URL · date · what it says (quote) · which H/U it bears on · confidence**.
"Not found" is a result and must be recorded too.

- [ ] **R1 Apple's own material:** `MLModel.load(contentsOf:configuration:completionHandler:)`, `MLModelConfiguration`
  (`functionName`, `optimizationHints`, `specializationStrategy`, `reshapeFrequency`), `MLModelAsset`, `MLComputePlan`,
  multifunction models (WWDC24 "Deploy machine learning and AI models on-device with Core ML"; WWDC23 "Improve Core
  ML integration with async prediction"; WWDC22 "Optimize your Core ML usage", which covers the device-specialization
  cache), the Core ML release notes and iOS 26/27 SDK diffs (any new caching or "prepare" API), Apple's File System
  Programming Guide / `Library/Caches` purge rules, and `NSURLIsPurgeableKey` / CacheDelete under low storage. The
  key question: **does Apple document where and for how long the specialised (device-compiled) model is cached, and
  what invalidates it?**
- [ ] **R2 Field reports:** Apple Developer Forums, the coremltools GitHub issues, Stack Overflow, and the llama.cpp,
  WhisperKit/argmaxinc, and apple/ml-stable-diffusion issues (these projects ship big ANE models, so they hit this).
  Search for "e5bundlecache", "e5rt", "ANECompilerService", "milhash", "`.functionName` property must be `nil`",
  "recompiles every launch", "specialization cache", "first load slow every time", "iOS 26 ANE compile",
  "iOS 27 Core ML". WhisperKit has written about "prewarm" and ANE specialization caching, so read its docs.
- [ ] **R3 What the framework binary says (time box 45 min):** strings from E5RT / Espresso / CoreML in this Mac's dyld
  shared cache, searched for `milhash`, `e5bundlecache`, `bundlecache`, `cache key`, `functionName must be nil`.
  Tool: `ipsw dyld str` (`brew install blacktop/tap/ipsw`, a local dev tool) or the REA MCP `open_binary` on the
  extracted framework. Use: string names only, to find the input names that go into the key and the logging
  subsystem/category names for Task 5. Don't patch or hook anything.
- [ ] **R4 How iOS purges caches:** when iOS deletes app `Library/Caches` content (storage thresholds, the
  `deleted`/CacheDelete daemon, "iPhone Storage almost full"), and whether it deletes files and leaves folders behind,
  which would explain N1's empty skeletons.
- [ ] **Step 5: Commit the research file.** For each H1–H7, add a line: "research supports / contradicts / is silent,
  sources #…".

```bash
git add docs/investigations/2026-10-2x-ane-cache-research.md
git commit -m "docs(investigation): ANE cache research with sources"
```

## Task 5: Mac control (E0)

The Mac's cache works across processes (F12), so it is the cheap place to learn what goes into the key and what
removes a payload. The OS differs (macOS 26.6.2 vs iOS 27.0.1), so a Mac result is a lead to test on the phone, never
proof about the phone.

**Files:**
- Create: `scripts/probes/mac_e0.sh`
- Create: `docs/investigations/evidence/2026-10-2x-ane-cache/e0-mac.md`

- [ ] **Step 1: Turn on debug logging and start a log stream (the founder runs the `sudo` line).**

```bash
sudo log config --subsystem com.apple.e5rt --mode level:debug   # plus any subsystem R3 found
log stream --level debug --style compact --predicate 'subsystem BEGINSWITH "com.apple.e5rt" OR subsystem == "com.apple.CoreML" OR process == "aned" OR process == "ANECompilerService"' > ~/egcm-evidence-raw/$(date +%F)/e0-stream.log &
```

- [ ] **Step 2: Write `mac_e0.sh`.** It runs one probe process on one function and records the cache state around it.

```bash
#!/bin/zsh
# usage: mac_e0.sh TAG MODEL_DIR FN [units]   e.g. mac_e0.sh e0a-1 /tmp/e0/b70.mlmodelc embed_text_16 ne
set -eu
TAG=$1; M=$2; FN=$3; UNITS=${4:-ne}; D=~/egcm-evidence-raw/$(date +%F)/$TAG; mkdir -p $D
C=~/Library/Caches/probe/com.apple.e5rt.e5bundlecache
REPO=$(git -C "$(dirname $0)" rev-parse --show-toplevel)
[ -x /tmp/probe ] || swiftc -O $REPO/scripts/probes/mac_load_probe.swift -o /tmp/probe
cd $REPO
uv run --exact --group convert python -m scripts.probes.e5_listing summarize $C > $D/pre.json 2>/dev/null || echo '{"keys":{}}' > $D/pre.json
stat -f '%i %m %z %N' $M/model.mil $M/weights/weight.bin > $D/model-stat.txt
/usr/bin/time -l /tmp/probe $M $UNITS $FN > $D/probe.txt 2> $D/time.txt
uv run --exact --group convert python -m scripts.probes.e5_listing summarize $C > $D/post.json
uv run --exact --group convert python -c "import json,sys; from scripts.probes.e5_listing import diff; a,b=(json.load(open(f)) for f in sys.argv[1:3]); print(json.dumps(diff(a,b)))" $D/pre.json $D/post.json | tee $D/diff.json
grep loaded $D/probe.txt
```

- [ ] **Step 3: E0a, the anatomy of a working cache.** First list `~/Library/Caches/probe` and save the listing, then
  remove that folder (it only holds this probe's own cache). Compile b70 once:
  `xcrun coremlcompiler compile Examples/PhotoSearchDemo/Model/b70/EmbeddingGemma2.mlpackage /tmp/e0/`, then rename
  the output to `/tmp/e0/b70.mlmodelc`. Run `mac_e0.sh e0a-1 … embed_text_16`. Wait 60 s and run `e0a-2`, same args.
  Do this 3 times (e0a-1…6). Record per run: load ms, new keys, milhash count, and whether `payload_lost` is non-empty.
  From the stream log, record the lines around each load (cache hit/miss wording, key or path names). **Expected for a
  working cache:** run 2 adds no key and loads in about 3.6 s.
- [ ] **Step 4: Find out where the ANE program lives (N2).**
  `sudo fs_usage -w -f filesys -e /tmp/probe 2>/dev/null | grep -iE 'e5|ane|hwx' > ~/egcm-evidence-raw/$(date +%F)/e0-fsusage.txt`
  during one cold run, plus `aned`'s and `ANECompilerService`'s file writes (`sudo fs_usage -w -f filesys aned
  ANECompilerService`). Record every path written under a cache folder.
- [ ] **Step 5: E0b, one variable at a time.** Start each row from a warm state (repeat the baseline run until it is
  warm), change exactly one thing, then run twice. Three repetitions per row.

| Row | Change | Command difference |
|---|---|---|
| b1 | none (control) | — |
| b2 | `touch` mtime of `model.mil` | `touch /tmp/e0/b70.mlmodelc/model.mil` |
| b3 | same bytes, new path | `cp -R /tmp/e0/b70.mlmodelc /tmp/e0/b70-copy.mlmodelc`, load the copy |
| b4 | same path, new inode | `cp -R` to a tmp folder, `rm -rf` the original, `mv` the copy back |
| b5 | `computeUnits` all vs ne | units arg `ne` → `cpu` → `ne` (does `ne` stay warm?) |
| b6 | 1-function package without `functionName` | build `t16` with `extract_functions.py`; needs a no-`functionName` flag in `mac_load_probe.swift` (add `NOFN=1`: skip `c.functionName = fn`) |
| b7 | two loads of the same function in one process | `fns = embed_text_16,embed_text_16` |
| b8 | executable renamed (`/tmp/probe2`) | shows that the cache root is per executable, so `probe2` is cold |

- [ ] **Step 6: Write `e0-mac.md`.** One table of every run (row, rep, ms, keys added, payload lost), the key-input
  findings stated only as far as the rows show them, and the matching log lines (trimmed). Commit it with `mac_e0.sh`.

## Task 6: Phone E1, the decisive minimal matrix

Before each phone launch the phone is unlocked and the probe is in front (the script launches it). Every row is
repeated 3 times (r1–r3), and every run is reported.

- [ ] **Step 1: E1a (H1, H3).** With ≥ 8 GB free and the probe freshly installed, run
  `phone_run.sh e1a-r1-A -fns embed_text_16`, then immediately `phone_watch.sh e1a-r1-w 2 8` (a listing every 2 min
  for 16 min, app not running), then `phone_run.sh e1a-r1-B -fns embed_text_16`. Repeat for r2 and r3. Between
  repetitions, uninstall and reinstall the probe so each repetition starts cold. Record:
  - A: cold ms; did the post listing show a bundle **with payload** (`bundles_with_payload` ≥ 1, `H14.e5` present)?
  - watch: does the payload disappear, and at which minute?
  - B: ms; new key or the same one?
- [ ] **Step 2: Read E1a with the decision tree.**
  - Payload present after A, still present at minute 16, B warm (≤ ~10 s) → no fault at this scale; go to E1b
    (longer gaps) and E1d (9 functions).
  - Payload present after A, gone during the watch → H1 confirmed at this scale; go to E1e (H2) and E2.
  - Payload never present after A (skeleton right after a clean exit) → the compile process doesn't keep it; go
    straight to E2 (logs of A) and E0's fs_usage to compare.
  - Payload present but B makes a new key → H3; go to E1c and the rows of E0b that showed a key change, now on the
    phone.
- [ ] **Step 3: E1b (U2, gaps):** the same A/B with B at +10 min, +1 h, and after a reboot (the founder restarts the
  phone; unlock it before B). 3 repetitions each where time allows. A reboot repetition = 1 per hour of waiting, so do
  r1 and run the others overnight if needed.
- [ ] **Step 4: E1c (H5):** E1a with `-model t16 -noFunctionName -fns embed_text_16`. Then `-api sync`.
- [ ] **Step 5: E1d (scale):** E1a with all 9 b70 functions (`-fns embed_text_16,…,embed_image_70_wide`), one A, a
  watch for 20 min, one B. 3 repetitions (about 30 min of compiling each; watch the thermal readings).
- [ ] **Step 6: E1e (H2, storage pressure):** only if the founder agrees to temporarily fill the phone (for example,
  by recording a long 4K video) down to < 2 GB free. Repeat E1a. Compare against E1a.
- [ ] **Step 7: Save the evidence.** For each row, add one line to `docs/investigations/evidence/2026-10-2x-ane-cache/e1-phone.md`:
  tag, since_last_s, free_mb, thermal, load ms, cache_post totals, diff. Commit after each row, not at the end.

## Task 7: Logs at a miss and at a `functionName` failure (E2, U3)

- [ ] **Step 1: Find the failing launch.** Right after the first phone launch that misses the cache (an E1 B that is
  cold, or an A whose payload vanished), the founder runs:

```bash
sudo log collect --device-udid 00008110-0014594C0A98401E --last 20m --output ~/egcm-evidence-raw/$(date +%F)/e2-<tag>.logarchive
log show ~/egcm-evidence-raw/$(date +%F)/e2-<tag>.logarchive --style compact --predicate \
 'process == "ANECompilerService" OR process == "aned" OR process == "deleted" OR process == "cachedeleted" OR subsystem CONTAINS[c] "coreml" OR subsystem CONTAINS[c] "e5rt" OR process == "PhotoSearchDemo"' \
 > ~/egcm-evidence-raw/$(date +%F)/e2-<tag>.txt
```

  Add any subsystems that R3 and E0 identified. If debug lines are missing, record that iOS doesn't persist them
  without a logging profile, and try `log config` on the device via the developer logging profile (R1 lists it if
  Apple documents one).
- [ ] **Step 2: Extract the evidence.** Lines mentioning the key, the cache path, a purge/delete, `milhash`,
  `functionName`, or the compile start/end, each with a timestamp. Trim them to < 5k tokens and save as
  `evidence/…/e2-<tag>.txt`.
- [ ] **Step 3: E2b, reproduce H6 on purpose.** Run `phone_run.sh e2b-A -fns embed_text_16` (warm entry made), then
  remove only the payload of that entry. Use the probe with a new `-wipePayload` flag: delete the regular files under
  `Library/Caches/<bundle id>/com.apple.e5rt.e5bundlecache/*/*/*.bundle`, keep the folders, and exit. Then run
  `phone_run.sh e2b-B -fns embed_text_16`. If B fails with the `functionName` error, H6 is confirmed. Its fix
  candidate is F-c in Task 9. Three repetitions. Add the flag to `Probe.swift`:

```swift
if args.contains("-wipePayload") {
  let root = URL.cachesDirectory.appending(path: "\(Bundle.main.bundleIdentifier!)/com.apple.e5rt.e5bundlecache")
  var removed = 0
  for case let f as URL in FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey])! {
    if f.path.contains(".bundle/"), (try? f.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true {
      try? FileManager.default.removeItem(at: f); removed += 1
    }
  }
  r.loads.append(Load(fn: "wipePayload", attempt: 1, ok: true, ms: removed, footprintMB: 0, peakMB: 0, thermal: 0, freeDiskMB: freeDiskMB()))
}
```

  Put this before the compile step in `runProbe()`, and pass no `-fns` on the wipe launch.
- [ ] **Step 4: Commit** the trimmed logs and an `e2-logs.md` that says which lines support or contradict which H.

