"""The s5 comparisons, as pure functions over what egcm-parity wrote and what s0 stored (no I/O, no Swift).

- `token_ids_identical`: every fixture text's ids equal s0's, exactly (and no text missing on either side).
- `max_pixel_abs_diff`: 99th percentile of |swift - reference| (0-1 units) over the pixel samples of the
  decode-identical inputs only (ADR-0004): the PNG fixtures against s0's stored bucket pixels and the synthetic PNGs
  against torch's uint8 antialiased bicubic. The JPEG number includes the decoder and is recorded, not gated.
  If any image lands in a different bucket than the reference, the pixels are not comparable and the metric is inf.
- `min_cosine`: minimum over every text vector and every image vector at both budgets (s0's `emb`, `emb_bucket`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import numpy as np

from egcm.metrics import cosine_rows
from stages.s5_swift.export_fixtures import sample_indices

U8 = np.float32(1 / 255)


def id_mismatches(got: Mapping[str, list[int]], ref: Mapping[str, list[int]]) -> list[str]:
    """Text ids whose token ids differ from the reference, or that only one side has. Sorted."""
    missing = set(ref) ^ set(got)
    differ = {k for k in set(ref) & set(got) if list(got[k]) != list(ref[k])}
    return sorted(missing | differ)


def reference_sample(u8_hwc: np.ndarray, samples: int) -> np.ndarray:
    """The reference uint8 HxWx3 image sampled like egcm-parity: `[3, H, W]` flattened, `sample_indices`."""
    chw = np.ascontiguousarray(np.asarray(u8_hwc, np.uint8).transpose(2, 0, 1)).reshape(-1)
    return chw[sample_indices(chw.size, samples)]


def sha256_hwc(u8_hwc: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(u8_hwc, np.uint8).tobytes()).hexdigest()


def pixel_diffs(got: Iterable[int], ref: np.ndarray) -> np.ndarray:
    """|got - ref| in 0-1 units (float32), as the processor scales: u8 * float32(1/255)."""
    g, r = np.asarray(list(got), np.int16), np.asarray(ref, np.int16)
    if g.shape != r.shape:
        raise ValueError(f"pixel samples differ in length: {g.shape} vs {r.shape}")
    return np.abs(g - r).astype(np.float32) * U8


def p99(parts: list[np.ndarray]) -> float:
    """99th percentile over all values combined; inf when there are none (the gate then fails visibly)."""
    values = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    return float(np.percentile(values, 99)) if values.size else float("inf")


class PixelTally:
    """Collects pixel comparisons per group ("png", "jpeg", "synthetic") and bucket or size mismatches."""

    def __init__(self, samples: int) -> None:
        self.samples = samples  # the `samples` egcm-parity used
        self.diffs: dict[str, list[np.ndarray]] = {"png": [], "jpeg": [], "synthetic": []}
        self.exact: dict[str, list[int]] = {k: [0, 0] for k in self.diffs}  # [bit-exact, total]
        self.mismatches: list[str] = []

    def add(self, group: str, key: str, got: Mapping, ref_bucket: str, ref_u8: np.ndarray) -> None:
        h, w = ref_u8.shape[:2]
        if (got["bucket"], got["height"], got["width"]) != (ref_bucket, h, w):
            self.mismatches.append(f"{key}: swift {got['bucket']} {got['height']}x{got['width']}, "
                                   f"reference {ref_bucket} {h}x{w}")
            return
        self.diffs[group].append(pixel_diffs(got["pixel_sample"], reference_sample(ref_u8, self.samples)))
        self.exact[group][0] += got["sha256"] == sha256_hwc(ref_u8)
        self.exact[group][1] += 1

    def metric(self) -> float:
        """The gated number: p99 over PNG + synthetic, or inf if any bucket differs anywhere."""
        if self.mismatches:
            return float("inf")
        return p99(self.diffs["png"] + self.diffs["synthetic"])

    def details(self) -> dict:
        out = {f"{g}_pixel_p99": p99(v) if v else None for g, v in self.diffs.items()}
        out |= {f"{g}_pixel_max": float(max(d.max() for d in v)) if v else None for g, v in self.diffs.items()}
        out["bit_exact"] = {g: f"{e}/{n}" for g, (e, n) in self.exact.items()}
        out["bucket_mismatches"] = self.mismatches
        return out


def cosine_summary(ids: list[str], got: np.ndarray, ref: np.ndarray, worst: int = 5) -> dict:
    """min and mean cosine, and the `worst` lowest ids."""
    cos = cosine_rows(np.asarray(got, np.float64), np.asarray(ref, np.float64))
    order = np.argsort(cos)[:worst]
    return {"min": float(cos.min()), "mean": float(cos.mean()), "n": len(ids),
            "worst": {ids[i]: float(cos[i]) for i in order}}


def summarize(parity: Mapping, ref_texts: Mapping[str, Mapping], ref_images: Mapping[int, Mapping[str, Mapping]],
              formats: Mapping[str, str], ref_synthetic: Mapping[str, Mapping]) -> tuple[dict, dict]:
    """(metrics, details) for the s5 report.

    parity: egcm-parity's JSON. ref_texts: id -> {"ids", "emb"} (s0). ref_images: budget -> id -> {"bucket", "u8"
    (HxWx3), "emb"} (s0's bucket path). formats: image id -> "png" or "jpeg". ref_synthetic: "<name>_<budget>" ->
    {"bucket", "u8"} (torch's resize). Every reference text and image must be in `parity`; extras are errors too.
    """
    texts = parity["texts"]
    bad_ids = id_mismatches({k: v["ids"] for k, v in texts.items()}, {k: v["ids"] for k, v in ref_texts.items()})
    tids = sorted(set(texts) & set(ref_texts))
    text_cos = {kind: cosine_summary(ids, np.array([texts[i]["emb"] for i in ids]),
                                     np.array([ref_texts[i]["emb"] for i in ids]))
                for kind, ids in _by(tids, lambda i: ref_texts[i].get("kind", "text")).items()}

    tally, image_cos, missing = PixelTally(int(parity["samples"])), {}, []
    for budget, refs in ref_images.items():
        keys = {f"{i}_{budget}": i for i in refs}
        missing += sorted(k for k in keys if k not in parity["images"])
        ids = sorted(i for k, i in keys.items() if k in parity["images"])
        for i in ids:
            group = formats[i]
            tally.add(group, f"{i}_{budget}", parity["images"][f"{i}_{budget}"], refs[i]["bucket"], refs[i]["u8"])
        for group, gids in _by(ids, lambda i: formats[i]).items():
            image_cos[f"{budget}_{group}"] = cosine_summary(
                gids, np.array([parity["images"][f"{i}_{budget}"]["emb"] for i in gids]),
                np.array([refs[i]["emb"] for i in gids]))
    extra = sorted(set(parity["images"]) - {f"{i}_{b}" for b, refs in ref_images.items() for i in refs})
    for key, ref in ref_synthetic.items():
        if key not in parity["synthetic"]:
            missing.append(key)
            continue
        tally.add("synthetic", key, parity["synthetic"][key], ref["bucket"], ref["u8"])

    mins = [s["min"] for s in (*text_cos.values(), *image_cos.values())]
    complete = not missing and not extra and bool(text_cos) and bool(image_cos)
    metrics = {
        "token_ids_identical": not bad_ids and bool(tids),
        "max_pixel_abs_diff": tally.metric() if not missing else float("inf"),
        "min_cosine": min(mins) if complete else float("nan"),
    }
    details = {"token_id_mismatches": bad_ids, "texts_compared": len(tids), "missing": missing, "unexpected": extra,
               "pixels": tally.details(), "cosine": {"text": text_cos, "image": image_cos},
               "model_revision": parity.get("model_revision"), "compute_units": parity.get("compute_units")}
    return metrics, details


def _by(ids: list[str], key) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for i in ids:
        out.setdefault(key(i), []).append(i)
    return out
