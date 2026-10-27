# ADR-0010: one package per image budget

**Status:** accepted, 2026-10-26 (founder chose per-budget packages, ADR-0009's decision). Changes what
`[s4_compress] max_download_mb` measures; its threshold (360) is unchanged.

## Context

ADR-0009 measured that loading any function of a multifunction package costs memory in proportion to the whole
package's `model.mil`. A phone uses one image budget only (`ImageBudget.auto`: 280 on A17 Pro and newer and on
Macs, 70 otherwise), yet today it downloads, compiles and loads a package with all 14 functions.

| package | device download | warm load per function | warm peak (Mac) | cold peak (Mac) |
|---|---|---|---|---|
| all 14 functions | 358.7 MB | 6.5-7 s | 969 MB | ~2.0 GB |
| text + budget 70 (9) | 343.1 MB | 3.6-3.8 s | not measured (a recompile) | 1,961 MB |
| text + budget 280 (9) | 352.8 MB | 3.6-3.8 s | 699 MB | 2,019 MB |

## Decision

s4 still builds, evaluates and gates the 14-function package exactly as before (ADR-0007). Then it splits the
winner into one package per budget, each holding the four text functions and that budget's five image functions,
and one device archive per budget.

- `max_download_mb` is the **larger** of the two device archives: what any one phone downloads. Threshold 360 MB,
  unchanged.
- s4 checks that the split changed nothing: every function of each budget package gives the same output as the
  14-function package on the same input (`.cpuOnly`, bitwise).
- s5 runs the Swift package and Python Core ML on the two budget packages (the bytes a phone gets), so its gates
  cover the split end to end. s6 bundles both and benchmarks the phone's own budget with its package (amended 2026-10-27: compiling both budgets on one iPhone 13 Pro Max made iOS evict the first compiles; see the s6 README).

## Published layout (the contract between Python, Swift and Roll)

Each budget is today's layout, in its own folder `b<budget>/` at the repo root:

    b70/manifest.json                 ArchiveManifest, plus "image_budget": 70
    b70/EmbeddingGemma2.aar           LZFSE Apple Archive: EmbeddingGemma2.mlpackage/** and tokenizer/ (as before)
    b70/files.json                    the plain layout (s6, Python users): files.json, EmbeddingGemma2.mlpackage/,
    b70/EmbeddingGemma2.mlpackage/**  tokenizer/
    b70/tokenizer/...
    b280/...                          the same for 280

- Inside every archive and plain layout the package is still named `EmbeddingGemma2.mlpackage`.
- `manifest.json` gains `"image_budget": <70|280>` (optional, so an old 14-function manifest still parses). When
  present, `functions` must include the four text functions and that budget's five image functions (not the other
  budget's). Without it, all 14 are required, as before.
- Swift, `.huggingFace(repo, revision)`: loads `b<resolved budget>/manifest.json` and its archive. The install
  folder is `<repo>/<revision>-b<budget>/`; `modelRevision` is still `<revision>` (callers that store vectors
  combine it with `imageBudget`, as Roll does).
- Swift, `.file(folder)`: uses `folder/b<budget>/` when that folder exists, else `folder` itself (a 14-function
  layout, e.g. s5's contract model). `modelRevision` is `file-<sha256 prefix of that folder's files.json>` (or
  manifest).
- `.installed(folder, ...)` is unchanged: the caller passes the layout of the budget it loads.
- `defaultManifestSHA256` becomes one pin per budget.

## Consequences

- Each phone downloads 343 or 353 MB instead of 359 MB, and its warm peak drops from ~970 MB to ~700 MB.
- The first compile still peaks at about 2 GB (ADR-0009's `max_cold_compile_peak_mb`).
- A phone that changes budget (not possible with `.auto`; possible with an explicit budget) downloads the other
  package too.
- s7 uploads two archives and two plain layouts.
