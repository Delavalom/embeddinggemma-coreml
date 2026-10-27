# E1 phone runs (probe v2). Raw files: ~/egcm-evidence-raw/2026-10-27/<tag>/ (pre/post listings, disk, probe.json)

Free space: `mac` = ideviceinfo `AmountDataAvailable` (the space actually free; it matches Settings). `app` =
`volumeAvailableCapacityForImportantUsage`, which includes space iOS can reclaim. Smoke run: app 26,511 MB vs mac
16,773 MB, so the two differ by about 9.7 GB. The handoff's free-disk numbers (F6, F9) are the `app` kind; actual free
space was lower.

| tag | args | since last (s) | free mac MB pre→post | thermal | load ms (fn) | peak MB | cache after (keys/bundles/with files) | payload lost | notes |
|---|---|---|---|---|---|---|---|---|---|
| smoke | b70 cpu text_16 | — | 16,773→16,021 | 0→0 | 14,825 | 1,198 | 1/1/1 | — | compile .mlpackage→.mlmodelc 11.0 s; a CPU load also writes an E5 bundle |
| e1a-r1-A | b70 ne text_16 | 28 | 16,211→16,601 | 0→0 | **46,414** (cold) | 1,375 | 3/2/2 | — | first ANE compile of text_16 on this install |
| e1a-r1-w | 9 listings, 2 min apart, app closed | — | 16,602→14,740→16,401 | — | — | — | 3/2/2 every time | none | free space moved by ±1.9 GB with our app closed (another process) |
| e1a-r1-B | b70 ne text_16 | 999 | 16,379→16,389 | 0→0 | **3,697** (warm) | 578 | 3/2/2, no key added | none | same entry reused 16.6 min later |
| e1a-r2-A | fresh install, b70 ne text_16 | — | 17,510→16,422 | 0→2 | **48,362** (cold) | 1,315 | 2/1/1 | — | key 9D2AEBFD |
| e1a-r2-w | 8 listings, 2 min apart, app closed | — | 16,423→14,739 (min) | — | — | — | 2/1/1 every time | none | |
| e1a-r2-B | b70 ne text_16 | 848 | 14,722→16,004 | 0→0 | **3,230** (warm) | 577 | no key added | none | |
| e1a-r3-A | fresh install, b70 ne text_16 | — | 16,601→15,846 | 0→1 | **47,839** (cold) | 1,315 | 2/1/1 | — | key B85C2D2E |
| e1a-r3-w | 8 listings, 2 min apart, app closed | — | 15,847→18,138→16,381 | — | — | — | 2/1/1 every time | none | |
| e1a-r3-B | b70 ne text_16 | 849 | 16,380→16,354 | 0→0 | **3,297** (warm) | 577 | no key added | none | |

## E1a result (3 of 3 rounds, ≥ 14.7 GB actually free throughout)

- Cold compile of `embed_text_16`: 46.4 / 48.4 / 47.8 s. Relaunch 14–17 min later: **3.7 / 3.2 / 3.3 s**, peak 577–578 MB.
- No compiled file lost in 25 listings over 3 × ~16 min; no key added by any relaunch.
- **The key differs per install, not per launch:** the same bytes of b70 got key 0883DD1D (r1), 9D2AEBFD (r2) and
  B85C2D2E (r3), one per fresh install (new container, new compiled-model path and inodes), and the relaunch within an
  install reused it every time. This explains the handoff's two keys for `embed_text_16` (072165CC before run 11, a
  separate install, and 10F6E2EA in run 11).
- So, with plenty of free space, on iOS 27.0.1: H1 (files removed with no pressure) and H3 (key changes per launch) were
  not seen at this scale (1 function, ~16 min). Still to do: H2 directly (low free space), longer gaps and a reboot
  (E1b), and 9 functions (E1d).

## App update (install over the r3 install, data kept) — one run

| tag | args | since last (s) | free mac MB | load ms | key | notes |
|---|---|---|---|---|---|---|
| e1e-update | b70 ne text_16 | 533 | 17,951→17,732 | **46,173 (cold)** | 2 keys added (new milhash + bundle); the old entry kept its files | data container moved CDA56F26… → A99077F9…; `b70.mlmodelc` files unchanged (same inodes, mtimes) |

The key changed after an update that changed only the container path and the app binary. The model files were byte-
and inode-identical. Which of the two (path or binary) feeds the key: test with `-from tmp` (same binary, new path).
Whether an App Store update moves the container the way `devicectl install` does: open.

## E1e — low free space, one variable (founder approved 14:4x)

| tag | step | free mac MB | cache (bundles / with files) | load ms | notes |
|---|---|---|---|---|---|
| e1e-r1-fill | `-fill 16200` (non-purgeable, Documents/filler.bin) | 17,730→1,536 | 2/2 | — | app's "important usage" free: 12,163 MB, so about 10.6 GB was purgeable |
| e1e-r1-w | 7 listings, 1 min apart, app closed | 1,536→1,312→**10,592** (14:56–14:57) | 2/2 throughout | — | iOS purged ~9 GB of *other* data; filler intact (16.98 GB); our files spared |
| e1e-r1-fill2 | `-fill 9024` | 10,572→2,255 | 2/2 | — | app free 3,474 → only ~1.2 GB purgeable left |
| e1e-r1-w2 | 8 listings, 1 min apart, app closed | 2,255→1,938→2,331→3,017 | **2/2 → 0/2 between 15:02:05 and 15:03:06** | — | iOS deleted the files of both our finished entries with our app not running |
| e1e-r1-B | b70 ne text_16 | 3,010→2,494 | 1/2 after | **45,910 (cold)** | same key (none added); recompiled into it; no `functionName` error |
| e1e-r2-w | 10 listings, 1 min apart, app closed, filler kept | 2,494→2,249 (min) | 1/1 throughout | — | no purge at ≥ 2.25 GB free in 10 min |
| e1e-r2-B | b70 ne text_16 | 2,246→2,247 | 1/1 | **3,144 (warm)** | |
| e1e-r3-w | 10 listings, filler kept | 2,247→**1,873** (15:18:59, other activity)→1,473→3,076 | **1/1 → 0/1 between 15:18:59 and 15:19:59** | — | purge with our app closed, at ~1.9 GB free |
| e1e-r3-B | b70 ne text_16 | 3,043→2,523 | 1/1 after | **45,700 (cold)** | same key; no `functionName` error |
| e1e-unfill | `-unfill` (removed 25,216 MB) | 2,523→27,742 | 1/1 | — | |

So far, low free space: our files were deleted in 2 of 3 rounds (r1, r3, both when free space went below ~1.9 GB), with
2 more deletions in yesterday's logs (11:33, 11:41). They were not deleted in r2, where free space stayed ≥ 2.25 GB
for 10 min. Every relaunch after a deletion was cold (45.9, 45.7 s); every relaunch without one was warm (3.1 s).
| e1e-r4-fill | `-fill 26240` (15:28:42–15:31:14Z+4h) | 27,743→2,049 | **1/1 → 0/1 during the fill** (lost 268 MB, key 370ED008) | — | app "important" free 28,359→2,246: almost nothing else purgeable left |
| e1e-r4-w | 10 listings, 1 min apart | 2,049→1,937 | 0/1 throughout | — | |
| e1e-r4-B | b70 ne text_16 | 1,936→1,396 | 1/1 after | **101,618 (cold)** | thermal 0→2; 2.2× slower than the other cold compiles (one run: note, not a finding yet); no `functionName` error |
| e1e-r4-unfill | removed 26,240 MB | 1,396→27,639 | 1/1 | — | |

## E1e result

- **Low free space deletes our compiled files: 3 of 3 deliberate rounds where free space went below ~2.0 GB** (r1 at
  15:03, log-attributed to iOS CacheDelete/APFS purge; r3 at ~1.9 GB; r4 during the fill to 2.05 GB), with our app
  not loading anything. Not deleted in r2 (≥ 2.25 GB for 10 min) or in E1a (≥ 14.7 GB, 3 × 16 min).
- Every relaunch after a deletion compiled cold (45.9, 45.7, 101.6 s); every relaunch without one was warm (3.1–3.7 s).
- iOS purged other purgeable data first (9.5 GB at 14:56 left ours alone); ours went when little else was left.
- `functionName` error: 0 of 4 cold relaunches against an emptied entry (r1, r3, r4, plus e1e-update). H6 is not
  supported so far; the error's trigger stays open (U3).
