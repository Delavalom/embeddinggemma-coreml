# s2_coreml_fp32

The rewrite converted to Core ML at FP32: every entry point of spec section 7 in one multifunction `.mlpackage`
with shared weights, run on the CPU (`CPU_ONLY`) on s0's inputs and compared with s0's vectors.

Run: `just stage 2` (the `convert` group). Needs fresh passing s0 and s1. About an hour on a busy M1 Pro: 14
conversions (0.5 to 2 minutes each), 14 merges (0.5 to 1 minute), then 200 texts and 400 images on the CPU.

## Input

- The HF snapshot (`egcm.config.HF_SNAPSHOT`), loaded strictly through `egcm.weights` (`egcm.functions.load_towers`).
- `artifacts/s0_reference/`: `texts.npz`, `images_{70,280}.npz`, `image_token_layout.json`.

## What it does

1. `egcm.parts.build`: each function in `egcm.functions.FUNCTIONS` is converted in its own process
   (`python -m egcm.parts convert`: loads only the weights it needs, traces, frees the torch module, converts at
   FP32, saves the part, exits with its memory and temporary files), then merged into the package in its own process
   (`python -m egcm.parts merge`: package so far + that part, `save_multifunction`, no torch), and the part is
   deleted. Resumable: `build.json` records the code hash, merged functions and package sha, so a rerun continues
   where the last one stopped. Guards: a child exits if its physical footprint passes 9 GB or swap grows by more
   than 6 GB; no step starts if it would leave less than 25 GB free. Measured 2026-10-23: conversion peaks at about
   2.6 GB (text) / 3.3 GB (image), a merge at about 4 GB; disk peaks at package + part + temporary copy (≈ 4.5 GB).
2. Checks the I/O contract of every function (`contract_problems`); a mismatch raises.
3. Compiles the package once and loads each function with `CompiledMLModel(function_name=...)` on `CPU_ONLY`.
4. Texts: padded with the config's `pad_id` to the smallest of 16/32/64/128, int32 mask. Images: s0's bucket pixels.
5. Deletes the package and the compiled model (not used downstream; s3 converts again at FP16). Set
   `EGCM_KEEP_PACKAGES=1` to keep them in `artifacts/s2_coreml_fp32/`.

## I/O contract (fixed; the Swift runtime codes against it)

| Function | Inputs | Output |
|---|---|---|
| `embed_text_{16,32,64,128}` | `input_ids` int32 [1, L], `attention_mask` int32 [1, L] | `embedding` fp32 [1, 768], L2-normalized |
| `embed_image_{70,280}_{tall,portrait,square,landscape,wide}` | `pixels` fp32 [1, 3, H, W] in [0, 1] | `embedding` fp32 [1, 768] |

Default function: `embed_text_16`.

## Output

`artifacts/s2_coreml_fp32/report.json`: metrics `min_cosine_text`, `min_cosine_image`; details `sizes_mb` (each
part and the merged package: the merged size ≈ one copy of the weights shows the deduplication worked),
`compiled_mb`, per-function cosines with the worst fixtures, `contract_problems` (empty).

## Gate

`gates.toml [s2_coreml_fp32]`: `min_cosine_text >= 0.9999`, `min_cosine_image >= 0.9999`.

## Known failure modes

- **An op does not convert** (einsum pattern, gather dtype): `tests/test_convertible.py` and `tests/test_convert.py`
  convert the tiny model the same way; reproduce there first.
- **Shape-derived ints in the trace** fail conversion (coremltools rejects the `int` ops). Every size must be fixed at
  construction (see egcm/model).
- **The merged package is about the sum of the parts**: deduplication needs bit-identical constants. A per-function
  difference in a shared weight (for example a different fusion) breaks it; compare `sizes_mb`.
- **Out of memory or disk.** On 2026-10-23 an in-process build (all parts converted in one process, merged 5 at a
  time) reached 16 GB of memory and ran the shared disk out of space (coremltools' temporary packages stayed until
  garbage collection). Hence one process per part and per merge. A guard exit says "memory guard"; a disk refusal
  says how much it needed. Rerunning resumes.
