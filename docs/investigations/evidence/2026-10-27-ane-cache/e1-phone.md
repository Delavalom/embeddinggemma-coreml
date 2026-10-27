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
