"""Export the small parity fixtures the Swift unit tests read (committed under swift/Tests/.../Fixtures/).

    uv run --exact --group convert python -m stages.s5_swift.export_fixtures [--s0 <artifacts/s0_reference>]

- `text_ids.json`: s0's token ids for every fixture text (prompt, bos and eos included).
- `pixels.json`: for 20 images (4 per bucket shape, PNG and JPEG) and both budgets, the bucket s0 chose and a
  deterministic sample of the bucket image s0 fed the model: the uint8 [3, H, W] tensor flattened, values at
  `sample_indices(3*H*W, SAMPLES)`, base64. Also the sha256 of the full uint8 HxWx3 image, to count exact matches.

The Swift test computes the 99th percentile of |swift - s0| over the samples, in 0-1 units, against
`[s5_swift].max_pixel_abs_diff`. Committed files stay well under 2 MB.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from egcm.buckets import BUDGETS, choose_bucket
from egcm.fixtures import S0_DIR, load_images, load_texts
from egcm.paths import HF_REVISION, ROOT

OUT_DIR = Path("swift/Tests/EmbeddingGemmaTests/Fixtures")
SAMPLES = 8192
SHAPES = ("tall", "portrait", "square", "landscape", "wide")


def sample_indices(n: int, samples: int) -> np.ndarray:
    """`(k * n) // samples` for k < samples: evenly spread, strictly increasing, no fixed stride to alias with."""
    if n <= samples:
        return np.arange(n, dtype=np.int64)
    return (np.arange(samples, dtype=np.int64) * n) // samples


def shape_of(row: dict) -> str:
    return choose_bucket(row["height"], row["width"], BUDGETS[0]).name


def pick_images(rows: list[dict], per_shape: int = 4) -> list[dict]:
    """Per bucket shape, in file order: up to half PNG and half JPEG, then any remaining images of that shape."""
    picked = []
    for shape in SHAPES:
        of_shape = [r for r in rows if shape_of(r) == shape]
        png = [r for r in of_shape if r["file"].endswith(".png")][: per_shape // 2]
        jpg = [r for r in of_shape if not r["file"].endswith(".png")][: per_shape - len(png)]
        chosen = png + jpg
        chosen += [r for r in of_shape if r not in chosen][: per_shape - len(chosen)]
        picked += sorted(chosen, key=rows.index)
    return picked


def image_entry(u8_hwc: np.ndarray, bucket: str, samples: int = SAMPLES) -> dict:
    h, w, _ = u8_hwc.shape
    chw = np.ascontiguousarray(u8_hwc.transpose(2, 0, 1)).reshape(-1)
    return {"bucket": bucket, "height": h, "width": w,
            "sha256": hashlib.sha256(np.ascontiguousarray(u8_hwc).tobytes()).hexdigest(),
            "sample_b64": base64.b64encode(chw[sample_indices(chw.size, samples)].tobytes()).decode()}


def export(s0: Path, out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    texts = load_texts()
    with np.load(s0 / "texts.npz") as z:
        rows = [{"id": t["id"], "kind": t["kind"], "text": t["text"], "ids": z[f"ids_{t['id']}"].tolist()}
                for t in texts]
    text_path = out / "text_ids.json"
    text_path.write_text(json.dumps({"hf_revision": HF_REVISION, "texts": rows}, ensure_ascii=False) + "\n")

    images = pick_images(load_images())
    entries = []
    npz = {b: np.load(s0 / f"images_{b}.npz") for b in BUDGETS}
    try:
        for r in images:
            budgets = {}
            for b in BUDGETS:
                z = npz[b]
                bucket = str(z["bucket"][list(z["order"]).index(r["id"])])
                if bucket != choose_bucket(r["height"], r["width"], b).name:
                    raise AssertionError(f"{r['id']}@{b}: s0 chose {bucket}")
                budgets[str(b)] = image_entry(z[f"pixels_{r['id']}"], bucket)
            entries.append({"id": r["id"], "file": r["file"], "height": r["height"], "width": r["width"],
                            "budgets": budgets})
    finally:
        for z in npz.values():
            z.close()
    pixel_path = out / "pixels.json"
    pixel_path.write_text(json.dumps({"hf_revision": HF_REVISION, "samples": SAMPLES,
                                      "sample_rule": "uint8 [3,H,W] flattened; index k -> (k * 3*H*W) // samples",
                                      "images": entries}, indent=1) + "\n")
    return text_path, pixel_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s0", type=Path, default=S0_DIR, help="s0_reference artifacts folder")
    ap.add_argument("--out", type=Path, default=ROOT / OUT_DIR)
    a = ap.parse_args()
    for p in export(a.s0, a.out):
        print(f"wrote {p} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
