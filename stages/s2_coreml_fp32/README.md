# s2_coreml_fp32

The rewrite converted to Core ML at FP32: every entry point of spec section 7 in one multifunction `.mlpackage`
with shared weights, run on the CPU (`CPU_ONLY`) on s0's inputs and compared with s0's vectors.

Run: `just stage 2` (the `convert` group). Needs fresh passing s0 and s1. About an hour on a busy M1 Pro: 14
conversions (1 to 3 minutes each), 3 merges, then 200 texts and 400 images on the CPU.

## Input

- The HF snapshot (`egcm.config.HF_SNAPSHOT`), loaded strictly through `egcm.weights` (`egcm.functions.load_towers`).
- `artifacts/s0_reference/`: `texts.npz`, `images_{70,280}.npz`, `image_token_layout.json`.

## What it does

1. For each function in `egcm.functions.FUNCTIONS`: build the torch module (`module_for`), trace it, convert with
   `egcm.convert.convert` (ML program, iOS18 / macOS15, FP32), save it as a part. Every 2 parts are merged into the
   growing package with `MultiFunctionDescriptor` + `save_multifunction` and deleted. `save_multifunction` writes a
   temporary copy of the weights, so disk peaks at about package + 2 parts + temporary weights + new package
   (≈ 7.5 GB) instead of 14 parts (≈ 16 GB). Free space is checked before every part and merge.
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
- **Out of disk** (`DiskSpaceError`, or `ENOSPC` in a merge): one FP32 part is about 1.1 GB and the merge needs a
  temporary copy of the weights. A batch of 5 ran out on 2026-10-23 with about 17 GB free on a shared disk; the
  batch is now 2. Delete `artifacts/s2_coreml_fp32/parts` and any `*.merging.mlpackage` before rerunning.
