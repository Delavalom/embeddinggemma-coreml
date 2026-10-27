# The phone's own logs: iOS purged the Core ML cache files (run 11, 2026-10-27; times AST)

Source: `sudo log collect --device-udid … --start "2026-10-27 11:30:00"` → `~/egcm-evidence-raw/2026-10-27/e2-run11.logarchive`
(557 MB, outside the repo). Read with `/usr/bin/log show --info --debug --style compact` (zsh's builtin `log` shadows
it). Our app container: `/private/var/mobile/Containers/Data/Application/1847FC54-…`; the benchmark process was
PhotoSearchDemo pid 704.

## Event 1, 11:41:38–39 (during the benchmark launch)

```
11:41:38.337 deleted   Purging /private/var/mobile to Desirable (1506167193 bytes)
11:41:38.347 deleted   Purge: goal 2576976281 freespace 1070809088
11:41:38.417 kernel    (apfs) disk1s8 reached the desired amount to delete (1884454912 > 1514584473)
11:41:38.417 kernel    (apfs) disk1s8 found 25 purgeable victims... deleting them.
11:41:38.417 kernel    (apfs) disk1s8 skipping purge of ino 105430282 (... mapped writable 128)
11:41:39.047 kernel    (apfs) disk1s8 Purged 1121705984 bytes in 14 files
11:41:39.048 kernel    (apfs) disk1s8 Purged 24576 bytes in 5 files
11:41:39.048 UserEventAgent FSEvent …e5bundlecache/24A446/{4E9ADFC6,689CA2CE,9FC80FAC,B4B2A25C,10F6E2EA}…/H14.bundle/H14.e5
11:41:39.048 UserEventAgent FSEvent …/{same 4 image keys, A4CC3941 (image_70_tall), 10F6E2EA (text_16)}…/<fn>_ane/model.anehash
11:41:39.061 deleted   Purge Result: 1083531264 bytes … freespace: 2154340352
```

## Event 2, 11:33:25–26 (during compile run 11, while it compiled embed_image_70_landscape)

```
11:33:25.890 deleted   Purging /private/var/mobile to Desirable (924739993 bytes)
11:33:26.258 kernel    (apfs) disk1s8 Purged 787271680 bytes in 6 files
11:33:26.328 kernel    (apfs) disk1s8 Purged 370323456 bytes in 3 files
11:33:26.182 FSEvent   3FA0D95F (text_128) …/embed_text_128_bnns/bnns_program.bnnsir
11:33:26.282 FSEvent   3FA0D95F (text_128) H14.e5; 887120C0 (text_32) and B1EB96F5 (text_64) model.anehash
11:33:26.383 FSEvent   3FA0D95F (text_128) model.anehash; A4CC3941 (image_70_tall) H14.e5
11:33:26.339 deleted   Purge Result: 1283907584 measured bytes
```

## All "Purging /private/var/mobile to Desirable" events in the archive (11:30–12:05)

11:33:25 (needs 0.92 GB) · 11:41:38 (1.51 GB) · 11:43:35 (1.66 GB) · 11:44:28 (1.69 GB) · 11:47:53 (2.07 GB) ·
11:50:13 (2.07 GB) · 11:52:52 (1.64 GB) · 11:54:23 (1.51 GB) · 12:01:43 (1.26 GB). Eight of them fall during runs
that compiled (run 11 compile 11:13–11:36, benchmark 11:37–11:50, relaunch from 11:51).

## What this establishes (facts)

1. The payload of finished cache entries (`H14.e5`, `model.anehash`, `bnns_program.bnnsir`) was deleted by iOS's
   CacheDelete purge (`deleted` → APFS purge of files marked purgeable), at the moment free space was below its goal.
   The FSEvents on our files fall in the same millisecond as the kernel's "Purged … files" lines.
2. Purges began **while our own compiles were running**: compile run 11 lost text-function payloads that the same
   run had compiled minutes earlier (11:33). The benchmark launch then found entries with no files and recompiled
   (handoff F7), which led to more purges.
3. Goal at 11:41: 2,576,976,281 bytes free (about 2.58 GB). Free space was 1.07 GB before the purge and 2.15 GB after.
4. Correction to handoff F2: keys 887120C0 (text_32), B1EB96F5 (text_64), 3FA0D95F (text_128) and 10F6E2EA (text_16)
   already existed at 11:33–11:41. The 11:51–11:53 times in the listing are last-modified times, so this listing doesn't
   show "a new key per launch".

## Still open

- Whether the payload survives when free space stays above the goal: E1a, now that the phone has 9 GB free.
- At what free space iOS *starts* a purge. The goal (2.58 GB) is known; the trigger isn't yet. Read it from
  `apfs_lowdisk_check_volume` / `processLowDiskVolume` lines.
- Why `embed_text_16` has two keys (072165CC at 11:07, before run 11 started; 10F6E2EA in run 11): one per install
  (s6 reinstalls the app) or one per launch? Test it in E1.
- Whether the `functionName` error comes from loading against an entry with no files (H6, E2b).
