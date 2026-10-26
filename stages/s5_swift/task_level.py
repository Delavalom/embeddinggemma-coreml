"""ADR-0008's s5 comparisons, as pure functions (no I/O, no models): one reference per question.

- **Runtime parity** (`min_cosine_runtime`): Swift's vector vs Python Core ML's, same package, same compute units,
  on Swift's exact inputs (token ids and bucket images). Every text and image vector.
- **Task level** (`min_cosine`, `min_recall_at_10`, `min_p5_spearman_top50`, `min_t2i_top1_in_top10`,
  `max_i2i_label_p10_drop`, as s4, `egcm.ranking.metrics`): Swift's vectors against the FP reference *on Swift's
  pixels*: s0's `emb_bucket` where Swift's bucket image is byte-identical to s0's, else the FP32 rewrite on Swift's
  image (stages/s5_swift/fp_reference.py). Texts: s0's `emb` (token ids are gated identical).
- **Recorded only**: the same metrics against s0's PIL-decoded reference (what s5 gated before ADR-0008), and the
  decoder's own cost: the FP reference on Swift's pixels against s0's.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from egcm import ranking
from egcm.metrics import cosine_rows
from stages.s5_swift.compare import cosine_summary

GATED = ("min_cosine", "min_recall_at_10", "min_p5_spearman_top50", "min_t2i_top1_in_top10", "max_i2i_label_p10_drop")
MIN_IMAGES = 50  # egcm.ranking's top-50 Spearman needs at least this many images per budget
RECORDED = ("min_cosine_text", "min_cosine_image", "min_p1_cosine_image", *GATED)


def runtime(parity: Mapping, py: Mapping[str, np.ndarray], kinds: Mapping[str, str],
            formats: Mapping[str, str]) -> tuple[float, dict]:
    """(min cosine, details) of Swift vs Python Core ML over every text and image in `parity`.

    py: "text_<id>" and "image_<id>_<budget>" -> raw vector. kinds: text id -> kind; formats: image id -> png/jpeg.
    A vector missing on the Python side makes the minimum NaN (the gate then fails) and is listed."""
    groups: dict[str, list[tuple[str, list, np.ndarray]]] = {}
    missing = []
    for k, t in sorted(parity["texts"].items()):
        if f"text_{k}" in py:
            groups.setdefault(f"text_{kinds[k]}", []).append((k, t["emb"], py[f"text_{k}"]))
        else:
            missing.append(f"text_{k}")
    for k, e in sorted(parity["images"].items()):
        image_id, budget = k.rsplit("_", 1)
        if f"image_{k}" in py:
            groups.setdefault(f"image_{budget}_{formats[image_id]}", []).append((k, e["emb"], py[f"image_{k}"]))
        else:
            missing.append(f"image_{k}")
    summary = {name: cosine_summary([r[0] for r in rows], np.array([r[1] for r in rows]),
                                    np.array([r[2] for r in rows])) for name, rows in groups.items()}
    low = min((s["min"] for s in summary.values()), default=float("nan"))
    return (float("nan") if missing else low), {"by_group": summary, "missing": missing}


def image_reference(order: list[str], emb_bucket: np.ndarray, fp: Mapping[str, np.ndarray]) -> np.ndarray:
    """Per row of `order`: the FP vector computed on Swift's pixels if there is one, else s0's `emb_bucket` row."""
    return np.stack([np.asarray(fp[i], np.float32) if i in fp else emb_bucket[n] for n, i in enumerate(order)])


def swift_vectors(parity: Mapping, s0_texts: Mapping, s0_images: Mapping[int, Mapping]) -> tuple[dict, dict]:
    """egcm.ranking.metrics inputs with Swift's vectors as `emb` and s0's as the references, restricted to (and in s0's
    order of) what `parity` covers. s0_texts: {"order", "emb"}; s0_images: budget -> {"order", "emb_bucket",
    "emb_raw"}."""
    rows = [n for n, k in enumerate(s0_texts["order"]) if k in parity["texts"]]
    order = [s0_texts["order"][n] for n in rows]
    emb = np.array([parity["texts"][k]["emb"] for k in order], np.float32)
    ref = np.asarray(s0_texts["emb"])[rows]
    texts = {"order": order, "emb": emb, "ref": ref, "cos": cosine_rows(emb, ref)}
    images = {}
    for b, z in s0_images.items():
        rows = [n for n, k in enumerate(z["order"]) if f"{k}_{b}" in parity["images"]]
        order = [z["order"][n] for n in rows]
        emb = np.array([parity["images"][f"{k}_{b}"]["emb"] for k in order], np.float32)
        bucket, raw = np.asarray(z["emb_bucket"])[rows], np.asarray(z["emb_raw"])[rows]
        images[b] = {"order": order, "emb": emb, "ref_bucket": bucket, "ref_raw": raw, "cos": cosine_rows(emb, bucket)}
    return texts, images


def with_reference(images: Mapping[int, Mapping], refs: Mapping[int, np.ndarray]) -> dict:
    """The same image inputs with `ref_bucket` replaced (and `cos` recomputed)."""
    return {b: dict(v, ref_bucket=refs[b], cos=cosine_rows(v["emb"], refs[b])) for b, v in images.items()}


def worst(images: Mapping[int, Mapping], n: int = 5) -> dict:
    """The `n` lowest image cosines per budget, as in the s5 cosine details."""
    out = {}
    for b, v in images.items():
        order = np.argsort(v["cos"], kind="stable")[:n]
        out[str(b)] = {v["order"][i]: float(v["cos"][i]) for i in order}
    return out


def keep(m: Mapping) -> dict:
    """The recorded subset of an egcm.ranking.metrics result, plus its per-budget ranking numbers."""
    return {k: m[k] for k in RECORDED} | {"vs_bucket": m["vs_bucket"]["by_budget"], "vs_raw": m["vs_raw"]["by_budget"]}


def task_level(texts: Mapping, images: Mapping[int, Mapping], fp_ref: Mapping[int, np.ndarray],
               fp_computed: Mapping[int, list[str]]) -> tuple[dict, dict]:
    """(gated metrics, details) for the task level, the s0 comparison and the decoder's cost.

    texts / images: `swift_vectors` output (reference s0). fp_ref: budget -> FP reference aligned with
    images[b]["order"] (`image_reference`). fp_computed: budget -> ids whose reference is the FP32 rewrite."""
    on_swift = with_reference(images, fp_ref)
    gated = ranking.metrics(texts, on_swift)
    vs_s0 = ranking.metrics(texts, images)
    s0_texts = dict(texts, emb=texts["ref"], cos=cosine_rows(texts["ref"], texts["ref"]))
    decoder_images = {b: dict(v, emb=fp_ref[b], cos=cosine_rows(fp_ref[b], v["ref_bucket"])) for b, v in images.items()}
    decoder = ranking.metrics(s0_texts, decoder_images)
    details = {
        "vs_fp_on_swift_pixels": keep(gated) | {"worst": worst(on_swift)},
        "vs_s0": keep(vs_s0) | {"worst": worst(images)},
        "decoder": {"min_cosine_image": decoder["min_cosine_image"],
                    "min_p1_cosine_image": decoder["min_p1_cosine_image"],
                    "min_recall_at_10": decoder["min_recall_at_10"],
                    "min_p5_spearman_top50": decoder["min_p5_spearman_top50"],
                    "worst": worst(decoder_images)},
        "fp_reference": {str(b): {"fp32_rewrite": len(fp_computed.get(b, [])),
                                  "s0_emb_bucket": len(images[b]["order"]) - len(fp_computed.get(b, []))}
                         for b in images},
    }
    return {k: gated[k] for k in GATED}, details


def kinds_and_formats(texts: list[dict], images: list[dict]) -> tuple[dict, dict]:
    """Fixture rows -> (text id -> kind, image id -> "png" / "jpeg")."""
    return ({t["id"]: t["kind"] for t in texts},
            {r["id"]: "png" if r["file"].lower().endswith(".png") else "jpeg" for r in images})
