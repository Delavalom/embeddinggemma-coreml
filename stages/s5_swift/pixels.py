"""Swift's exact bucket images (egcm-parity --pixels, ADR-0008): read back and checked against parity.json.

egcm-parity writes each fixture's uint8 bucket image as `<dir>/<id>_<budget>.png` and records its sha256 (the uint8
HxWx3 layout s0 stores). Everything s5 feeds to Python Core ML or the FP reference comes through `swift_pixels`, so a
dump that is not byte-identical to what the Swift package fed its model raises instead of being compared.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from egcm.buckets import BUCKETS, Bucket
from egcm.fixtures import to_unit
from stages.s5_swift.compare import sha256_hwc


class PixelDumpError(RuntimeError):
    """A dumped bucket image is missing, has the wrong size, or is not the image Swift hashed."""


def bucket_for(name: str, budget: int) -> Bucket:
    for b in BUCKETS:
        if b.name == name and b.budget == budget:
            return b
    raise PixelDumpError(f"no bucket {name!r} at budget {budget}")


def swift_pixels(entry: Mapping, folder: Path, key: str) -> np.ndarray:
    """The uint8 HxWx3 bucket image Swift used for parity entry `key` ("<id>_<budget>"), sha256-checked."""
    p = Path(folder) / f"{key}.png"
    if not p.is_file():
        raise PixelDumpError(f"{p} is missing: egcm-parity must run with --pixels {folder}")
    with Image.open(p) as im:
        if im.mode != "RGB":
            raise PixelDumpError(f"{p}: mode {im.mode}, expected RGB")
        u8 = np.asarray(im, dtype=np.uint8)
    if u8.shape != (entry["height"], entry["width"], 3):
        raise PixelDumpError(f"{p}: shape {u8.shape}, parity.json says {entry['height']}x{entry['width']}")
    if sha256_hwc(u8) != entry["sha256"]:
        raise PixelDumpError(f"{p}: sha256 differs from the one egcm-parity recorded for {key}")
    return u8


def model_input(u8_hwc: np.ndarray) -> np.ndarray:
    """`pixels` [1, 3, H, W] float32 in 0-1, as egcm.fixtures.load_bucket_pixels makes it from s0's pixels."""
    return np.ascontiguousarray(to_unit(u8_hwc).transpose(2, 0, 1)[None])
