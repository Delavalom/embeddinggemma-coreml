# The `functionName` error, from the phone's logs (2026-10-27, AST)

Archives (outside the repo): `~/egcm-evidence-raw/2026-10-27/e2-run11.logarchive` (11:30–12:05) and
`e1d-fn-error.logarchive` (from 15:44). Read with `/usr/bin/log show --info --debug`, processes `kernel`, `aned`,
`ANECompilerService`, `PhotoSearchDemo`. Every one of the 6 errors in these archives is listed.

## Failure A: the compiler process is killed by the memory manager (3 of 6)

```
15:50:57.899 kernel memorystatus: memorystatus_perform_idle_demotion() timed out stuck process 1251 [ANECompilerService], moving to idle band
15:51:10.224 kernel memorystatus: ANECompilerService [1251] exceeded mem limit: InactiveHard 6 MB (fatal)
15:51:10.225 kernel memorystatus: killing_specific_process pid 1251 [ANECompilerService] (per-process-limit 0 12s rf:low type:daemon) 8018KB
15:51:10.226 aned  Could not create connection to com.apple.ANECompilerService : error=… NSCocoaErrorDomain Code=4097
15:51:10.432 PhotoSearchDemo [com.apple.e5rt] E5RT encountered an STL exception.
15:51:10.539 PhotoSearchDemo [com.apple.coreml] `MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program.
```
The same kernel → aned → E5RT → coreml sequence: 11:30:33.442 (ANECompilerService pid 636) and 11:38:40.343 (pid 699).
15:51 context (e1d-r1-A): 26.9 GB free (CacheDelete query in our process at 15:50:25: `freespace 26869760000`),
thermal state 2 (serious) from the first function on; 6th of 9 functions (`embed_image_70_portrait`), 45.4 s into
the load, first attempt. So disk space is not involved in failure A.

## Failure B: the compiler returns an error (3 of 6)

```
11:44:44.424 ANECompilerService ERROR: model=<private> … lErr=Error Domain=com.apple.appleneuralengine.compiler Code=1
11:44:44.425 aned  Compilation failed: error=Error Domain=com.apple.appleneuralengine.compiler Code=1
11:44:44.647 PhotoSearchDemo E5RT encountered an STL exception.
11:44:44.747 PhotoSearchDemo `.functionName` property must be `nil` …
```
Also 11:48:01.195 → 11:48:01.498 and 11:50:21.224 → 11:50:21.492 (the benchmark's final failure, handoff F7). The
compiler process (pid 736) was alive throughout. Its reason text is `<private>` (redacted).
Timing: each B failure came 8–16 s after iOS started a low-disk purge ("Purging /private/var/mobile to Desirable":
11:44:28, 11:47:53, 11:50:13; e2-run11-ios-purge.md), during the benchmark launch at ~1–2 GB free.
**Hypothesis, not established:** the compiler fails when the disk runs out mid-compile. Needs the unredacted
reason (logging profile) or a low-disk compile test.

## What this establishes

1. The `functionName` message is Core ML's wrong report of a failed Neural Engine compile. It is not a problem with
   our `functionName` or package: the same function compiles fine on another attempt or launch (F8).
2. There are at least two causes: (A) iOS's memory manager kills `ANECompilerService` after demoting it to the idle
   band (InactiveHard 6 MB limit); (B) the compiler returns `com.apple.appleneuralengine.compiler Code=1`.
3. Also from e1d-fn-error at 15:50:25: the compiled ANE program is loaded from
   `/var/mobile/Library/Caches/com.apple.aned/24A446/ModelAssetsCache/com.delavalom.egcm.demo_…` (aned's cache, outside
   our app container: counted as System Data, not freed by uninstalling the app), with "on-device compiled model
   DataVault verification succeeded".

Open: why the compiler is demoted to idle while working ("timed out stuck process"); whether a retry after
failure A always succeeds; what Code=1's redacted reason is.
