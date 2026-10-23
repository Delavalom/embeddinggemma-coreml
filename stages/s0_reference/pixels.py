"""The pixel contract between our bucket images and the reference processor, in plain numpy.

s0 asserts, bit for bit, that the processor's patches equal `patchify(bucket_u8)` and its positions equal
`patch_positions(...)`, so the stored uint8 image is a complete record of what the model saw.
"""

from __future__ import annotations

import numpy as np

from egcm.buckets import PATCH
from egcm.fixtures import to_unit


def patchify(u8_hwc: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """uint8 HxWx3 -> float32 [(H/p)*(W/p), p*p*3]: rows of patches in row-major order, each flattened in
    (py, px, c) order. Same as upstream `convert_image_to_patches` applied to `to_unit(u8)` in CHW."""
    h, w, c = u8_hwc.shape
    if h % patch or w % patch:
        raise ValueError(f"{h}x{w} is not a multiple of the patch size {patch}")
    x = to_unit(u8_hwc).reshape(h // patch, patch, w // patch, patch, c).transpose(0, 2, 1, 3, 4)
    return x.reshape((h // patch) * (w // patch), patch * patch * c)


def patch_positions(height: int, width: int, patch: int = PATCH) -> np.ndarray:
    """int32 [(H/p)*(W/p), 2] of (x, y) patch coordinates, row-major, as the processor's meshgrid makes them."""
    hp, wp = height // patch, width // patch
    y, x = np.divmod(np.arange(hp * wp, dtype=np.int32), wp)
    return np.stack([x, y], axis=1).astype(np.int32)
