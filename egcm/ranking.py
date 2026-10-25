"""Task-level metrics of a Core ML package against s0 (s3, s4): cosines, text->image ranking, bucketing fidelity.

ADR-0006 and ADR-0007 (accepted 2026-10-25):
- Ranking is gated against the **bucketed FP reference**: s0's query vectors against s0's bucketed image vectors
  (`emb_bucket`, which s2 matches to 2e-8); the new side is the package's query vectors against its own vectors for
  the same bucketed images. That isolates what FP16 and compression cost.
- Bucketing is gated separately against s0's **exact-aspect** image vectors (`emb_raw`) at the task level: the
  exact-aspect top-1 image must stay in the package's top 10, and image->image category precision@10 must not drop.
  Text->image recall@10 against exact-aspect is recorded (`vs_raw`), not gated.
The 120 app queries (`kind=query`) rank the 200 images, separately at budget 70 and 280; gated values are the worst of
the two budgets.
"""

from __future__ import annotations

import numpy as np

from egcm.buckets import BUDGETS
from egcm.fixtures import load_images, load_texts
from egcm.metrics import recall_at_k, spearman_topk


def query_rows(order: list[str]) -> list[int]:
    kinds = {t["id"]: t["kind"] for t in load_texts()}
    return [i for i, k in enumerate(order) if kinds[k] == "query"]


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def top1_in_topk(*, ref, new, k: int = 10) -> float:
    """Share of queries whose top-1 doc under `ref=(queries, docs)` is among the top `k` under `new`."""
    top1 = np.argmax(_unit(ref[0]) @ _unit(ref[1]).T, axis=1)
    topk = np.argsort(-(_unit(new[0]) @ _unit(new[1]).T), axis=1, kind="stable")[:, :k]
    return float(np.mean([t in row for t, row in zip(top1, topk, strict=True)]))


def label_precision_at_k(docs: np.ndarray, labels: list[str], k: int = 10) -> float:
    """Image->image retrieval quality without ground-truth pairs: for every doc, the share of its `k` nearest other
    docs (cosine, itself excluded) with the same label; mean over docs."""
    s = _unit(docs) @ _unit(docs).T
    np.fill_diagonal(s, -np.inf)
    nn = np.argsort(-s, axis=1, kind="stable")[:, :k]
    lab = np.asarray(labels)
    return float(np.mean(lab[nn] == lab[:, None]))


def metrics(texts: dict, images: dict) -> dict:
    """texts / images as returned by egcm.evaluate.run_texts / run_images.

    Gated keys: `min_cosine_text`, `min_cosine_image`, `min_cosine` (both), `min_p1_cosine_image` (1st percentile,
    linear interpolation, over every image vector of both budgets), `min_recall_at_10` and `min_p5_spearman_top50`
    (against the bucketed FP reference), `min_t2i_top1_in_top10` and `max_i2i_label_p10_drop` (against exact-aspect).
    Per-budget values are under `vs_bucket` and `vs_raw`."""
    q = query_rows(texts["order"])
    q_ref, q_new = texts["ref"][q], texts["emb"][q]
    img_cos = np.concatenate([np.asarray(images[b]["cos"]) for b in BUDGETS])
    out: dict = {"min_cosine_text": float(np.min(texts["cos"])), "min_cosine_image": float(img_cos.min()),
                 "min_p1_cosine_image": float(np.percentile(img_cos, 1))}
    out["min_cosine"] = min(out["min_cosine_text"], out["min_cosine_image"])
    category = {r["id"]: r["category"] for r in load_images()}
    bucket, raw = {}, {}
    for b in BUDGETS:
        d_new, d_bucket, d_raw = images[b]["emb"], images[b]["ref_bucket"], images[b]["ref_raw"]
        bucket[b] = {"recall_at_10": recall_at_k(ref=(q_ref, d_bucket), new=(q_new, d_new), k=10),
                     "p5_spearman_top50": spearman_topk(ref=(q_ref, d_bucket), new=(q_new, d_new), k=50)}
        labels = [category[i] for i in images[b]["order"]]
        p10_ref, p10_new = label_precision_at_k(d_raw, labels), label_precision_at_k(d_new, labels)
        raw[b] = {"recall_at_10": recall_at_k(ref=(q_ref, d_raw), new=(q_new, d_new), k=10),
                  "p5_spearman_top50": spearman_topk(ref=(q_ref, d_raw), new=(q_new, d_new), k=50),
                  "t2i_top1_in_top10": top1_in_topk(ref=(q_ref, d_raw), new=(q_new, d_new), k=10),
                  "i2i_label_p10_ref": p10_ref, "i2i_label_p10": p10_new, "i2i_label_p10_drop": p10_ref - p10_new}
    out["vs_bucket"] = {"by_budget": bucket,
                        "min_recall_at_10": min(v["recall_at_10"] for v in bucket.values()),
                        "min_p5_spearman_top50": min(v["p5_spearman_top50"] for v in bucket.values())}
    out["vs_raw"] = {"by_budget": raw, "min_recall_at_10": min(v["recall_at_10"] for v in raw.values()),
                     "min_p5_spearman_top50": min(v["p5_spearman_top50"] for v in raw.values())}
    out["min_recall_at_10"] = out["vs_bucket"]["min_recall_at_10"]
    out["min_p5_spearman_top50"] = out["vs_bucket"]["min_p5_spearman_top50"]
    out["min_t2i_top1_in_top10"] = min(v["t2i_top1_in_top10"] for v in raw.values())
    out["max_i2i_label_p10_drop"] = max(v["i2i_label_p10_drop"] for v in raw.values())
    return out
