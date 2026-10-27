# E0 (Mac control): what goes into the E5 cache key — 2026-10-27, M1 Pro, macOS 26.6.2 (25G83)

Tool: `scripts/probes/mac_e0.sh` (one `/tmp/probe` process = `scripts/probes/mac_load_probe.swift`, one function,
`.cpuAndNeuralEngine`, cache listing before/after with `e5_listing`). Model: `xcrun coremlcompiler compile` of the b70
package → `/tmp/e0/b70.mlmodelc`. Function `embed_text_16`. Raw: `~/egcm-evidence-raw/2026-10-27/e0-*`.
The model bytes never change below. Only the named variable changes.

| Variable changed | Runs | Load ms | Keys added | Result |
|---|---|---|---|---|
| none: first load at /tmp/e0/b70.mlmodelc | 1 | 59,563 | 2 | cold |
| none: same path again | 3 (p2, p5, p8-baseline) | 3,991 / 3,963 / — | 0 | warm |
| **path + inodes + mtimes** (`cp -R` to a new folder) | 3 | 61,930 / 60,038 / 60,017 | 2 each | cold |
| same copies again | 3 | 3,938 / 3,817 / 4,164 | 0 | warm |
| **path only** (`mv`: inode and mtime unchanged, checked with `stat`) | 3 | 58,841 / 58,816 / 58,313 | 2 each | **cold** |
| **inodes only** (`cp -Rp` + `rm` + `mv` back: same path, mtimes preserved, new inodes) | 3 | 58,195 / 61,851 / 60,308 | 2 each | **cold** |
| inodes + mtimes, same path (`cp -R` + `rm` + `mv` back) | 1 | 58,711 | 2 | cold |
| `model.mil` mtime only (`touch`) | 1 | 3,773 | 0 | warm |
| `weights/weight.bin` mtime only (`touch`) | 2 | 3,870 / 3,822 | 0 | warm |

Result (Mac): the key includes the compiled model's **path** and its files' **identity (inode)**. File mtimes are not
in it (`model.mil`, `weight.bin`). Peak memory: cold 1,414–1,470 MB, warm 546 MB.

Phone, consistent so far (not yet one-variable): after `devicectl install` over the existing app (e1e-update), the
data container moved to a new path with the same inodes, and the load was cold with a new key. The phone check is
`-from tmp` (new path every launch, same binary).
