# Which free-space reading matches the phone's true free space (2026-10-27, 19:5x AST)

Probe launched 3 times with no functions (`-tag cap1..3`); `ideviceinfo -q com.apple.disk_usage` read just before and
4 s after each launch. Raw: `~/egcm-evidence-raw/2026-10-27/cap/`.

| Run | AmountDataAvailable (MB, before/after) | `volumeAvailableCapacity` (MB) | `volumeAvailableCapacityForImportantUsage` (MB) |
|---|---|---|---|
| cap1 | 12,265 / 12,267 | 12,460 | 27,523 |
| cap2 | 12,267 / 12,266 | 12,460 | 27,523 |
| cap3 | 12,282 / 12,280 | 12,460 | 27,523 |

- `volumeAvailableCapacity` is within 1.6 % of AmountDataAvailable (the reading that matches Settings and that the purge
  follows, `phone-baseline-1257.txt`). It did not change across the 3 reads while AmountDataAvailable moved 17 MB, so
  iOS may round or cache it.
- The important-usage reading counts 15 GB of purgeable data, our own compiled cache included, so it says nothing about
  how close the phone is to a purge. `EmbeddingGemma.prewarm`'s reserve therefore uses `volumeAvailableCapacity`
  (`ModelStore.realFreeSpace`).
