# Run 11 timeline: when the phone's cache payload disappeared (2026-10-27, times UTC; AST = UTC-4)

Source: `scripts/probes/e5_listing.py` on `~/egcm-evidence-raw/2026-10-27/listing-pre-uninstall.json` (devicectl
`--json-output`, 12:53 AST), plus the run 11 progress files in this folder.

| UTC | Event | Evidence |
|---|---|---|
| 15:13:53 | compile run 11 starts; free 4,745 MB | `compile-run11-9fn.json` |
| 15:36:54 | compile run 11 done, 9/9 compiled (cold: 62–325 s each); min free 3,151 MB; thermal serious | same |
| 15:37:04 | benchmark launch starts (loads functions again) | `benchmark-run11-failed.json` |
| **15:41:39** | **the folders of 4 cache entries (`embed_image_70_{landscape,wide,portrait,square}`, 4 different keys) all get the same mtime, to the second. All 4 hold no files now.** | listing (key folders 4E9ADFC6, 689CA2CE, 9FC80FAC, B4B2A25C) |
| 15:50:21 | benchmark fails: `functionName` error, 0 images | `benchmark-run11-failed.json`, `console-benchmark-run11.log` |
| 15:51:22 | relaunch; free at launch 2,411 MB, min 1,627 MB; text_16/32/64 cold (43/46/52 s) | `relaunch-after-9fn-compile-1153.json` |

Fact: 4 entries for 4 functions that compiled minutes apart were modified in the same second, during a launch, 4.5 min
after the compile process ended. A folder's mtime changes when entries inside it are removed. That fits one bulk
deletion (H1/H2) better than separate compiles finishing. It does not say who deleted them: the next step is the
phone's system log for 15:41 UTC (`log collect`), filtered for `deleted`, `cachedeleted`, `CacheDelete`, `e5rt`, and
`aned`.

All finished bundles, including the 9 completed ones: 0 files. Unfinished `.tmp.` bundles: 4 (3 empty, 1 with
408 MB: `embed_text_128`, killed mid-compile at 15:54).
