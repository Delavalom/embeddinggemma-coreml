"""Weight compression recipes for s4, applied to each single-function part before the multifunction merge.

coremltools 9 does not compress multifunction models, so every part is compressed on its own and the merge
deduplicates the results. That only works if the same weight compresses to the same bytes in every part, so every
recipe here is deterministic: linear quantization (scales from the weight itself). k-means palettization would need
scikit-learn or kmeans1d (neither is in the lock; adding one changes uv.lock and makes every stage stale).

A recipe quantizes two kinds of constants and leaves every other one FP16:
- conv weights (every projection of both towers is a 1x1 conv), except the patch embedding (C_in = 3, 0.6M
  parameters; per-block int4 cannot split it and coremltools skips it);
- the embedding table (262,144 x 512, the `gather` input of the text functions), set separately.
Position embeddings, RoPE tables, the image affixes, norm weights and biases stay FP16: they feed elementwise ops, are
small (29 MB over all 14 functions at FP16) and are what int4 would hurt most.

The Neural Engine compiler (M1 Pro, macOS 26) rejects some per-block int4 weights (ANECCompile() FAILED; Core ML
then cannot load that function of a multifunction package on CPU_AND_NE; ADR-0007 "ANE compile"): the vision
attention k and v projections (block 32 or 64) at every image budget, and the text tower inside the budget-280 image
functions (it compiles in the text functions and at budget 70). int8 compiles everywhere. So the int4 recipe keeps the
text tower (which also needs int8 for accuracy) and the vision k/v projections at int8.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import coremltools as ct
import coremltools.optimize.coreml as cto

# Constants with fewer elements are never quantized (coremltools' default threshold is 2048).
THRESHOLD = 2048
VISION_ATTN_KV = r"^vision_encoder_layers_\d+_attn_[kv]_weight"
TEXT_TOWER = r"^(?!vision_)"  # every conv weight that is not the vision tower's


def _int8() -> cto.OpLinearQuantizerConfig:
    return cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel",
                                       weight_threshold=THRESHOLD)


def _int4(block: int) -> cto.OpLinearQuantizerConfig:
    return cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", granularity="per_block",
                                       block_size=block, weight_threshold=THRESHOLD)


def _role(name: str, meta) -> str | None:
    """"conv" for a conv weight (not the patch embedding), "table" for a gather input, else None (stays FP16)."""
    ops = meta.child_ops
    if ops and all(o.op_type == "conv" and o.params_name_mapping.get("weight") == name for o in ops):
        return "conv" if meta.val.ndim == 4 and meta.val.shape[1] >= 32 else None
    if ops and all(o.op_type == "gather" for o in ops):
        return "table"
    return None


def configs(m: ct.models.MLModel, conv: cto.OpLinearQuantizerConfig, table: cto.OpLinearQuantizerConfig,
            overrides: tuple[tuple[str, cto.OpLinearQuantizerConfig], ...] = ()) -> dict:
    """{const name: config} for one model: `table` for the table, the first matching `overrides` regex (on the const
    name) or `conv` for conv weights. Constants not listed stay FP16."""
    out = {}
    for name, meta in cto.get_weights_metadata(m, weight_threshold=THRESHOLD).items():
        role = _role(name, meta)
        if role == "table":
            out[name] = table
        elif role == "conv":
            out[name] = next((c for rx, c in overrides if re.search(rx, name)), conv)
    return out


def _recipe(conv, table, overrides=()) -> Callable[[ct.models.MLModel], ct.models.MLModel]:
    def apply(m: ct.models.MLModel) -> ct.models.MLModel:
        names = configs(m, conv(), table(), tuple((rx, c()) for rx, c in overrides))
        return cto.linear_quantize_weights(m, cto.OptimizationConfig(op_name_configs=names))

    return apply


def _int4b32():
    return _int4(32)


# name -> (description, compress(mlmodel) -> mlmodel), smallest first. Measured sizes and accuracy: ADR-0007.
RECIPES: dict[str, tuple[str, Callable[[ct.models.MLModel], ct.models.MLModel]]] = {
    "vis4kv8_text8": ("int4 per-block (32) vision tower and table; vision attention k/v and the text tower int8 "
                      "per-channel (Neural Engine compiler); tables and norms FP16",
                      _recipe(_int4b32, _int4b32, ((VISION_ATTN_KV, _int8), (TEXT_TOWER, _int8)))),
    "int8_table4": ("int8 per-channel conv weights, int4 per-block (32) table; tables and norms FP16",
                    _recipe(_int8, _int4b32)),
    "int8": ("int8 per-channel conv weights and table; tables and norms FP16", _recipe(_int8, _int8)),
}

# Measure-only variants (s4 --measure), never stage candidates: "fp16" is the uncompressed FP16 package on the Neural
# Engine, the baseline that separates what FP16 on the Neural Engine costs from what compression costs (ADR-0006).
MEASURE_ONLY: dict[str, tuple[str, Callable[[ct.models.MLModel], ct.models.MLModel]]] = {
    "fp16": ("FP16, no weight compression (baseline)", lambda m: m),
}
