"""egcm.ranking: the task-level metrics s3 and s4 gate (ADR-0006, ADR-0007), on synthetic vectors."""

import numpy as np
import pytest

from egcm import ranking
from egcm.fixtures import load_images, load_texts


def fake(noise_raw: float, noise_new: float, seed: int = 0, cos: float = 1.0):
    """s0-shaped texts / images: the reference is `ref_bucket`; `ref_raw` adds bucketing noise, `emb` model noise."""
    rng = np.random.default_rng(seed)
    order = [t["id"] for t in load_texts()]
    q = rng.normal(size=(len(order), 16))
    texts = {"order": order, "ref": q, "emb": q + noise_new * rng.normal(size=q.shape), "cos": np.ones(len(order))}
    ids = [r["id"] for r in load_images()]
    images = {}
    for b in (70, 280):
        d = rng.normal(size=(len(ids), 16))
        images[b] = {"order": ids, "ref_bucket": d, "ref_raw": d + noise_raw * rng.normal(size=d.shape),
                     "emb": d + noise_new * rng.normal(size=d.shape), "cos": np.full(len(ids), cos)}
    return texts, images


def test_identical_vectors_pass_everything():
    m = ranking.metrics(*fake(0.0, 0.0))
    assert m["min_recall_at_10"] == 1.0 and m["min_p5_spearman_top50"] == 1.0
    assert m["min_t2i_top1_in_top10"] == 1.0 and m["max_i2i_label_p10_drop"] == 0.0
    assert m["min_cosine"] == m["min_cosine_image"] == m["min_p1_cosine_image"] == 1.0


def test_ranking_gates_are_against_the_bucketed_reference():
    m = ranking.metrics(*fake(noise_raw=0.5, noise_new=0.0))
    assert m["min_recall_at_10"] == 1.0 and m["min_p5_spearman_top50"] == 1.0  # gated: vs bucketed FP
    assert m["vs_raw"]["min_recall_at_10"] < 0.95  # bucketing cost: recorded, not in the gated keys
    assert m["min_t2i_top1_in_top10"] < 1.0


def test_model_noise_lowers_the_gated_ranking():
    m = ranking.metrics(*fake(noise_raw=0.0, noise_new=0.5))
    assert m["min_recall_at_10"] < 0.95 and m["min_p5_spearman_top50"] < 0.9


def test_p1_cosine_tolerates_two_outliers_the_minimum_does_not():
    texts, images = fake(0.0, 0.0)
    images[70]["cos"][:2] = 0.994  # two of 400 image vectors, as i141 / i161 in s3
    m = ranking.metrics(texts, images)
    assert m["min_cosine_image"] == pytest.approx(0.994)
    assert m["min_p1_cosine_image"] == pytest.approx(1.0)
    images[70]["cos"][:10] = 0.994  # ten of 400: the 1st percentile drops too
    assert ranking.metrics(texts, images)["min_p1_cosine_image"] == pytest.approx(0.994)


def test_top1_in_topk():
    q = np.eye(3)
    docs = np.eye(3)
    assert ranking.top1_in_topk(ref=(q, docs), new=(q, docs), k=1) == 1.0
    swapped = docs[[1, 0, 2]]  # queries 0 and 1 now find the other doc first
    assert ranking.top1_in_topk(ref=(q, docs), new=(q, swapped), k=1) == pytest.approx(1 / 3)
    assert ranking.top1_in_topk(ref=(q, docs), new=(q, swapped), k=3) == 1.0


def test_label_precision_excludes_the_doc_itself():
    docs = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
    assert ranking.label_precision_at_k(docs, ["a", "a", "b", "b"], k=1) == 1.0
    assert ranking.label_precision_at_k(docs, ["a", "b", "a", "b"], k=1) == 0.0
    assert ranking.label_precision_at_k(docs, ["a", "a", "b", "b"], k=3) == pytest.approx(1 / 3)


def test_label_drop_is_reference_minus_package():
    texts, images = fake(0.0, 0.0)
    for b in (70, 280):
        images[b]["emb"] = np.random.default_rng(1).normal(size=images[b]["emb"].shape)  # unrelated vectors
    m = ranking.metrics(texts, images)
    r = m["vs_raw"]["by_budget"][70]
    assert r["i2i_label_p10_drop"] == pytest.approx(r["i2i_label_p10_ref"] - r["i2i_label_p10"])
    assert m["max_i2i_label_p10_drop"] == max(v["i2i_label_p10_drop"] for v in m["vs_raw"]["by_budget"].values())


def test_s3_reports_every_gated_metric():
    pytest.importorskip("coremltools")
    from egcm.gates import gates
    from stages.s3_coreml_ane.run import GATED

    m = ranking.metrics(*fake(0.0, 0.0))
    assert set(gates("s3_coreml_ane")) == {*GATED, "min_ane_fraction"}
    assert all(k in m for k in GATED)
