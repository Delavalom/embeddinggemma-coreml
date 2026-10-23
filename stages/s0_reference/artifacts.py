"""Writers for the s0 output files. No model code here, so the formats can be tested in the convert group."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np


class NpzWriter:
    """Writes an .npz one array at a time (deflated), so large outputs never sit in memory together.

    Use as a context manager. The file appears atomically on a clean exit; on an exception the temp file is
    removed and nothing is written. `np.load` reads the result."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.zip: zipfile.ZipFile | None = None

    def __enter__(self) -> NpzWriter:
        self.zip = zipfile.ZipFile(self.tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True)
        return self

    def add(self, name: str, arr) -> None:
        with self.zip.open(f"{name}.npy", "w", force_zip64=True) as f:
            np.lib.format.write_array(f, np.asanyarray(arr), allow_pickle=False)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.zip.close()
        if exc_type is None:
            self.tmp.replace(self.path)
        else:
            self.tmp.unlink(missing_ok=True)


def write_keys(tensors: dict[str, tuple[list[int], str]], out: Path) -> Path:
    """keys.json: {key: [shape, dtype]} for every tensor in model.safetensors, sorted by key."""
    p = out / "keys.json"
    p.write_text(json.dumps({k: [list(s), d] for k, (s, d) in sorted(tensors.items())}, indent=1) + "\n")
    return p


def write_texts(out: Path, order: list[str], emb: np.ndarray, ids: dict[str, np.ndarray]) -> Path:
    """texts.npz: `order`, `emb` [N, 768] float32, `ids_<id>` int32 token ids."""
    with NpzWriter(out / "texts.npz") as w:
        w.add("order", np.array(order))
        w.add("emb", emb.astype(np.float32))
        for k in order:
            w.add(f"ids_{k}", ids[k].astype(np.int32))
    return out / "texts.npz"


def image_sequence(special: dict[str, int], soft_tokens: int) -> list[int]:
    """The token sequence s0 recorded for one image: bos, boi, soft_tokens image placeholders, eoi, eos."""
    return [special["bos"], special["boi"], *[special["image"]] * soft_tokens, special["eoi"], special["eos"]]


def write_layout(out: Path, special: dict[str, int], buckets: dict[str, dict], text_example: dict) -> dict:
    """image_token_layout.json. Each bucket entry has input_ids and tokens; checks them against image_sequence."""
    matches = all(b["input_ids"] == image_sequence(special, b["soft_tokens"]) for b in buckets.values())
    layout = {
        "special_token_ids": special,
        "image_sequence": "<bos> <|image> <|image|> x soft_tokens <image|> <eos>" if matches
                          else "DIFFERS per bucket: read input_ids",
        "matches_bos_boi_image_eoi_eos": matches,
        "text_example": text_example,
        "buckets": buckets,
    }
    (out / "image_token_layout.json").write_text(json.dumps(layout, indent=1, ensure_ascii=False) + "\n")
    return layout
