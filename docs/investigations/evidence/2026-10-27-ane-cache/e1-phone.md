# E1 phone runs (probe v2). Raw files: ~/egcm-evidence-raw/2026-10-27/<tag>/ (pre/post listings, disk, probe.json)

Free space: `mac` = ideviceinfo `AmountDataAvailable` (the space actually free; it matches Settings). `app` =
`volumeAvailableCapacityForImportantUsage`, which includes space iOS can reclaim. Smoke run: app 26,511 MB vs mac
16,773 MB, so the two differ by about 9.7 GB. The handoff's free-disk numbers (F6, F9) are the `app` kind; actual free
space was lower.

| tag | args | since last (s) | free mac MB pre→post | thermal | load ms (fn) | peak MB | cache after (keys/bundles/with files) | payload lost | notes |
|---|---|---|---|---|---|---|---|---|---|
| smoke | b70 cpu text_16 | — | 16,773→16,021 | 0→0 | 14,825 | 1,198 | 1/1/1 | — | compile .mlpackage→.mlmodelc 11.0 s; a CPU load also writes an E5 bundle |
