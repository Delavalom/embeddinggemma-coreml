# E1e: the phone's logs for the low-free-space test (2026-10-27, AST)

Source: `sudo log collect --start "2026-10-27 14:45:00"` → `~/egcm-evidence-raw/2026-10-27/e1e.logarchive` (outside the
repo); `/usr/bin/log show --info --debug`, processes `deleted`, `kernel`, `UserEventAgent` (FSEvents).

| Time | `deleted` | kernel (APFS, disk1s8 = user data) | FSEvents on our E5 cache |
|---|---|---|---|
| 14:56:52.803 | Purging /private/var/mobile to Desirable (862,923,161 bytes) | Purged 8,799,703,040 bytes in 14 files; 174,993,408 in 57; 30,797,824 in 21 | **none** |
| 14:56:54.340 | Purge Result: 9,544,372,224 measured bytes | | |
| 14:59:49.389 | Purging … to Desirable (678,390,169 bytes) | | **none** |
| 14:59:50.460 | Purge Result: 682,696,704 measured bytes | | |
| **15:03:00.420** | **Purging … to Desirable (640,481,689 bytes)** | | |
| 15:03:00.522 | | found 27 purgeable victims... deleting them | |
| **15:03:00.681** | | **Purged 691,716,096 bytes in 25 files** | |
| 15:03:00.696 | Purge Result: 734,040,064 measured bytes | | |
| **15:03:00.724** | | | **370ED008/E1A55BE1 and B85C2D2E/ECE6D764: `H14.e5`, `embed_text_16_ane/model.anehash`, `embed_text_16_bnns/bnns_program.bnnsir`** |
| 15:07:24–15:08:06 | | | the probe's launch B recompiles into the same key 370ED008 (`.tmp` → `H14.bundle`, new `bnnsir`, `H14.e5`) |

Facts:
1. With the app not running, iOS's purge deleted the compiled files of both our E5 entries (two bnnsir files are
   ~512 MB of the 692 MB in that purge).
2. Two earlier purges (9.5 GB and 683 MB) took other data and left ours. Ours went when little other purgeable data was
   left (the app's "important usage" free minus actual free: ~10.6 GB at 14:52, ~1.2 GB at 15:00).
3. The listing (e1-phone.md) shows the same: 2/2 bundles with files at 15:02:05, 0/2 at 15:03:06.
