"""Export the small parity fixtures the Swift unit tests read (committed under swift/Tests/.../Fixtures/).

    uv run --exact --group convert python -m stages.s5_swift.export_fixtures [--s0 <artifacts/s0_reference>]

- `text_ids.json`: s0's token ids for every fixture text (prompt, bos and eos included).
- `pixels.json`, `images`: for 20 images (4 per bucket shape, PNG and JPEG) and both budgets, the bucket s0 chose and a
  deterministic sample of the bucket image s0 fed the model: the uint8 [3, H, W] tensor flattened, values at
  `sample_indices(3*H*W, SAMPLES)`, base64. Also the sha256 of the full uint8 HxWx3 image, to count exact matches.

- `pixels.json`, `synthetic`: a fixed procedural pattern (`synthetic`, reproduced exactly in Swift) at
  `SYNTHETIC_SIZES`, resized to its bucket at both budgets by torch's uint8 antialiased bicubic (the kernel s0 used,
  which reproduces s0 bit for bit), sampled the same way. These are lossless inputs, so they test the resize alone.

JPEG decoding differs between ImageIO and PIL (libjpeg-turbo), so the fixture JPEGs cannot match s0 to 1/255 in
any Swift resize (docs/decisions/ADR-0004-pixel-gate-inputs.md). The pixel gate is therefore taken on the
decode-identical inputs (synthetic and PNG); the JPEG numbers are recorded.

- `pixels.json`, `large_synthetic`: the same pattern at photo sizes (12 MP, 48 MP, extreme aspect ratios; 9x to
  18x downscales), SHA-256 of torch's uint8 output only, so the files stay small.
- `embeddings.json`: s0's normalized 768-d vectors (float32, base64) for `EMB_TEXTS` and, at both budgets,
  `EMB_IMAGES` (bucket path): the end-to-end test's targets once s4 exists.

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
EMB_TEXTS = ("t000", "t001", "t002", "t003", "t120", "t121", "t122", "t199")  # queries, documents, the longest
EMB_IMAGES = ("i000", "i012", "i015", "i023", "i100", "i006")  # landscape, tall, square, wide, PNG tall, portrait
SHAPES = ("tall", "portrait", "square", "landscape", "wide")
# (height, width): every bucket at both budgets, downscale up to 6.75x, upscale (300x400, 100x2000 vertically),
# and the 280 portrait bucket's own size (both passes skipped).
SYNTHETIC_SIZES = ((1000, 1500), (1500, 1000), (777, 777), (2000, 700), (650, 1900), (300, 400), (480, 900),
                   (960, 672), (2268, 3024), (100, 2000))
# (height, width, budgets): large downscales, SHA only. 48 MP is checked by the release perf test.
LARGE_SYNTHETIC = ((3024, 4032, (70, 280)), (5000, 1200, (70,)), (1080, 7000, (70,)), (6048, 8064, (70, 280)))


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


def synthetic(height: int, width: int) -> np.ndarray:
    """uint8 HxWx3: diagonal gradients, 32-bit hash noise and a 5 px checkerboard in 37x29 tiles. Integer-only, so
    the Swift tests make the identical image (TestSupport.synthetic)."""
    y, x = (a.astype(np.uint32) for a in np.mgrid[0:height, 0:width])
    region = ((x // 37) + (y // 29)) % 3
    check = np.where(((x // 5) + (y // 5)) % 2 == 1, 255, 0).astype(np.uint32)
    out = np.empty((height, width, 3), np.uint8)
    for c in range(3):
        noise = ((x * np.uint32(73856093)) ^ (y * np.uint32(19349663)) ^ np.uint32(c * 83492791)) >> np.uint32(24)
        grad = (x + 2 * y + 60 * c) % 256
        out[..., c] = np.select([region == 0, region == 1], [grad, noise], check).astype(np.uint8)
    return out


def torch_resize(u8_hwc: np.ndarray, height: int, width: int) -> np.ndarray:
    """torch's uint8 bicubic + antialias, the kernel torchvision's resize runs in s0."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(u8_hwc)).permute(2, 0, 1)[None]
    r = torch.nn.functional.interpolate(t, size=(height, width), mode="bicubic", antialias=True)
    return r[0].permute(1, 2, 0).contiguous().numpy()


def image_entry(u8_hwc: np.ndarray, bucket: str, samples: int = SAMPLES) -> dict:
    h, w, _ = u8_hwc.shape
    chw = np.ascontiguousarray(u8_hwc.transpose(2, 0, 1)).reshape(-1)
    return {"bucket": bucket, "height": h, "width": w,
            "sha256": hashlib.sha256(np.ascontiguousarray(u8_hwc).tobytes()).hexdigest(),
            "sample_b64": base64.b64encode(chw[sample_indices(chw.size, samples)].tobytes()).decode()}


def b64_f32(v: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(v, dtype="<f4").tobytes()).decode()


def export_embeddings(s0: Path, out: Path) -> Path:
    kinds = {t["id"]: t for t in load_texts()}
    with np.load(s0 / "texts.npz") as z:
        order = list(z["order"])
        texts = [{"id": k, "kind": kinds[k]["kind"], "text": kinds[k]["text"],
                  "emb_b64": b64_f32(z["emb"][order.index(k)])} for k in EMB_TEXTS]
    rows = {r["id"]: r for r in load_images()}
    images = [{"id": i, "file": rows[i]["file"], "budgets": {}} for i in EMB_IMAGES]
    for b in BUDGETS:
        with np.load(s0 / f"images_{b}.npz") as z:
            order = list(z["order"])
            for e in images:
                e["budgets"][str(b)] = b64_f32(z["emb_bucket"][order.index(e["id"])])
    p = out / "embeddings.json"
    p.write_text(json.dumps({"hf_revision": HF_REVISION, "texts": texts, "images": images}, indent=1,
                            ensure_ascii=False) + "\n")
    return p


def export(s0: Path, out: Path) -> tuple[Path, ...]:
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
    synth = []
    for h, w in SYNTHETIC_SIZES:
        img, budgets = synthetic(h, w), {}
        for b in BUDGETS:
            bk = choose_bucket(h, w, b)
            budgets[str(b)] = image_entry(torch_resize(img, bk.height, bk.width), bk.name)
        synth.append({"height": h, "width": w, "budgets": budgets})
    large = []
    for h, w, budgets in LARGE_SYNTHETIC:
        img = synthetic(h, w)
        for b in budgets:
            bk = choose_bucket(h, w, b)
            r = torch_resize(img, bk.height, bk.width)
            large.append({"height": h, "width": w, "budget": b, "bucket": bk.name,
                          "sha256": hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest()})
    pixel_path = out / "pixels.json"
    pixel_path.write_text(json.dumps({"hf_revision": HF_REVISION, "samples": SAMPLES,
                                      "sample_rule": "uint8 [3,H,W] flattened; index k -> (k * 3*H*W) // samples",
                                      "images": entries, "synthetic": synth,
                                      "large_synthetic": large}, indent=1) + "\n")
    return text_path, pixel_path, export_embeddings(s0, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s0", type=Path, default=S0_DIR, help="s0_reference artifacts folder")
    ap.add_argument("--out", type=Path, default=ROOT / OUT_DIR)
    a = ap.parse_args()
    for p in export(a.s0, a.out):
        print(f"wrote {p} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
