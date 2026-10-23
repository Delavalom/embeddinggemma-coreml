"""Weight compression recipes for s4, applied to each single-function part before the multifunction merge.

coremltools 9 does not compress multifunction models, so every part is compressed on its own and the merge
deduplicates the results. That only works if the same weight compresses to the same bytes in every part, so every
recipe here is deterministic: linear quantization (scales from the weight itself). k-means palettization would need
scikit-learn or kmeans1d (neither is in the lock; adding one changes uv.lock and makes every stage stale).

The embedding table (262,144 x 512, the input of the text functions' `gather`) is the largest tensor; recipes can set
its precision separately through `op_type_configs={"gather": ...}`.
"""

from __future__ import annotations

from collections.abc import Callable

import coremltools as ct
import coremltools.optimize.coreml as cto

# Weights with fewer elements stay FP16 (norm weights, layer scalars, RoPE and position tables are not weights of a
# conv/gather or are small); coremltools' default threshold is 2048.
THRESHOLD = 2048


def _int8() -> cto.OpLinearQuantizerConfig:
    return cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel",
                                       weight_threshold=THRESHOLD)


def _int4(block: int) -> cto.OpLinearQuantizerConfig:
    return cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", granularity="per_block",
                                       block_size=block, weight_threshold=THRESHOLD)


def _recipe(weights: cto.OpLinearQuantizerConfig, table: cto.OpLinearQuantizerConfig):
    def apply(m: ct.models.MLModel) -> ct.models.MLModel:
        cfg = cto.OptimizationConfig(global_config=weights, op_type_configs={"gather": table})
        return cto.linear_quantize_weights(m, cfg)

    return apply


# name -> (description, compress(mlmodel) -> mlmodel), smallest first
RECIPES: dict[str, tuple[str, Callable[[ct.models.MLModel], ct.models.MLModel]]] = {
    "int4b32": ("int4 per-block (32) everywhere, table included", _recipe(_int4(32), _int4(32))),
    "int4b32_table8": ("int4 per-block (32) weights, int8 per-channel table", _recipe(_int4(32), _int8())),
    "int8": ("int8 per-channel everywhere", _recipe(_int8(), _int8())),
}
