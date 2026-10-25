"""A tiny stand-in model with the real I/O contract, so the Swift runtime can be tested before s4 exists.

    uv run --exact --group convert python -m stages.s5_swift.make_contract_model

Writes artifacts/s5_swift/contract/EmbeddingGemma2.mlpackage: one multifunction package with every entry point of
spec section 7 (`embed_text_{16,32,64,128}`: `input_ids`, `attention_mask` int32 [1, L]; `embed_image_<budget>_<shape>`:
`pixels` fp32 [1, 3, H, W]; all output `embedding` [1, 768]). Each function computes a fixed, easily recomputed
vector, `features @ W` with W[k, d] = sin((k + 1) * (d + 1) / 100):

- text: features = [sum(ids * mask) / 1e5, sum(mask) / 128, L / 128, 1]
- image: features = [mean R, mean G, mean B, H / 1000]

Three extra functions with text_16 inputs test output handling: `test_nan_text_16` (all NaN),
`test_fp16_text_16` (the text vector as FP16) and `test_4d_text_16` (the text vector shaped [1, 768, 1, 1]).

swift/Tests/EmbeddingGemmaTests/RuntimeTests.swift recomputes them (it skips when the package is missing). Not a
model of anything; a few KB.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

from egcm.buckets import BUCKETS
from egcm.paths import ARTIFACTS

OUT = ARTIFACTS / "s5_swift" / "contract" / "EmbeddingGemma2.mlpackage"
TEXT_LENGTHS = (16, 32, 64, 128)
W = np.stack([np.sin((k + 1) * (np.arange(768) + 1) / 100) for k in range(4)]).astype(np.float32)
TARGET = ct.target.macOS15


def text_program(length: int, variant: str = ""):
    @mb.program(input_specs=[mb.TensorSpec((1, length), dtype=types.int32),
                             mb.TensorSpec((1, length), dtype=types.int32)], opset_version=TARGET)
    def prog(input_ids, attention_mask):
        ids, m = mb.cast(x=input_ids, dtype="fp32"), mb.cast(x=attention_mask, dtype="fp32")
        s = mb.mul(x=mb.reduce_sum(x=mb.mul(x=ids, y=m), axes=[1], keep_dims=True), y=np.float32(1e-5))
        n = mb.mul(x=mb.reduce_sum(x=m, axes=[1], keep_dims=True), y=np.float32(1 / 128))
        const = np.array([[length / 128, 1.0]], np.float32)
        feats = mb.concat(values=[s, n, const], axis=1)
        if not variant:
            return mb.matmul(x=feats, y=W, name="embedding")
        emb = mb.matmul(x=feats, y=W)
        if variant == "nan":
            return mb.mul(x=emb, y=np.float32(np.nan), name="embedding")
        if variant == "fp16":
            return mb.cast(x=emb, dtype="fp16", name="embedding")
        return mb.reshape(x=emb, shape=[1, 768, 1, 1], name="embedding")

    return prog


def image_program(height: int, width: int):
    @mb.program(input_specs=[mb.TensorSpec((1, 3, height, width), dtype=types.fp32)], opset_version=TARGET)
    def prog(pixels):
        means = mb.reduce_mean(x=pixels, axes=[2, 3], keep_dims=False)  # [1, 3]
        feats = mb.concat(values=[means, np.array([[height / 1000]], np.float32)], axis=1)
        return mb.matmul(x=feats, y=W, name="embedding")

    return prog


def main() -> None:
    programs = {f"embed_text_{n}": text_program(n) for n in TEXT_LENGTHS}
    programs |= {b.function_name: image_program(b.height, b.width) for b in BUCKETS}
    programs |= {f"test_{v}_text_16": text_program(16, v) for v in ("nan", "fp16", "4d")}
    with tempfile.TemporaryDirectory() as tmp:
        desc = ct.utils.MultiFunctionDescriptor()
        for name, prog in programs.items():
            path = Path(tmp) / f"{name}.mlpackage"
            ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=TARGET,
                       compute_precision=ct.precision.FLOAT32).save(str(path))
            desc.add_function(str(path), src_function_name="main", target_function_name=name)
        desc.default_function_name = "embed_text_16"
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        ct.utils.save_multifunction(desc, str(OUT))
    print(f"wrote {OUT} with {len(programs)} functions")


if __name__ == "__main__":
    main()
