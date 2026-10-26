"""ADR-0008's s5 comparisons: runtime parity, the task level against the FP reference on Swift's pixels, and the
recorded s0 / decoder numbers; plus the pixel dump reader and the work lists of the two Python steps."""

import hashlib
import math

import numpy as np
import pytest
from PIL import Image

from egcm.fixtures import load_images, load_texts
from stages.s5_swift import fp_reference, pixels, run, runtime
from stages.s5_swift import task_level as tl

D = 8


def unit(seed: int, d: int = D) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=d)
    return (v / np.linalg.norm(v)).astype(np.float32)


def sha(u8: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(u8).tobytes()).hexdigest()


def fixture_ids(n_texts: int = 40, n_images: int = 60):
    """Real fixture ids (egcm.ranking reads kinds and categories from the fixtures), queries included."""
    texts, images = load_texts()[:n_texts], load_images()[:n_images]
    assert any(t["kind"] == "query" for t in texts)
    return texts, images


def s0_and_parity(texts, images, budgets=(70, 280)):
    """s0-like vectors and a parity file whose Swift vectors equal them."""
    s0_texts = {"order": [t["id"] for t in texts], "emb": np.stack([unit(n) for n in range(len(texts))])}
    s0_images = {b: {"order": [r["id"] for r in images],
                     "emb_bucket": np.stack([unit(1000 + b + n) for n in range(len(images))]),
                     "emb_raw": np.stack([unit(5000 + b + n) for n in range(len(images))])} for b in budgets}
    parity = {"texts": {k: {"ids": [2, 1], "emb": s0_texts["emb"][n].tolist()}
                        for n, k in enumerate(s0_texts["order"])},
              "images": {f"{k}_{b}": {"budget": b, "bucket": "square", "emb": z["emb_bucket"][n].tolist()}
                         for b, z in s0_images.items() for n, k in enumerate(z["order"])}}
    return s0_texts, s0_images, parity


def test_runtime_identical_is_one_and_groups_by_kind_budget_format():
    texts, images = fixture_ids(6, 4)
    _, _, parity = s0_and_parity(texts, images)
    py = {f"text_{k}": 3 * np.asarray(t["emb"]) for k, t in parity["texts"].items()}  # raw outputs: scale is fine
    py |= {f"image_{k}": np.asarray(e["emb"]) for k, e in parity["images"].items()}
    kinds, formats = tl.kinds_and_formats(texts, images)
    low, d = tl.runtime(parity, py, kinds, formats)
    assert low == pytest.approx(1.0) and d["missing"] == []
    assert {g.split("_")[0] for g in d["by_group"]} == {"text", "image"}
    assert all(g.startswith(("text_query", "text_document", "image_70_", "image_280_")) for g in d["by_group"])


def test_runtime_takes_the_worst_vector_and_missing_is_nan():
    texts, images = fixture_ids(4, 3)
    _, _, parity = s0_and_parity(texts, images)
    py = {f"text_{k}": np.asarray(t["emb"]) for k, t in parity["texts"].items()}
    py |= {f"image_{k}": np.asarray(e["emb"]) for k, e in parity["images"].items()}
    key = f"{images[1]['id']}_280"
    py[f"image_{key}"] = unit(77)
    kinds, formats = tl.kinds_and_formats(texts, images)
    low, _ = tl.runtime(parity, py, kinds, formats)
    assert low == pytest.approx(float(np.dot(unit(77), parity["images"][key]["emb"])), abs=1e-6)
    del py[f"text_{texts[0]['id']}"]
    low, d = tl.runtime(parity, py, kinds, formats)
    assert math.isnan(low) and d["missing"] == [f"text_{texts[0]['id']}"]


def test_image_reference_uses_fp_only_where_computed():
    emb = np.stack([unit(1), unit(2), unit(3)])
    ref = tl.image_reference(["a", "b", "c"], emb, {"b": unit(9)})
    np.testing.assert_array_equal(ref[0], emb[0])
    np.testing.assert_array_equal(ref[1], unit(9))
    np.testing.assert_array_equal(ref[2], emb[2])


def test_task_level_gates_against_the_fp_reference_and_records_s0_and_decoder():
    texts, images = fixture_ids()
    s0_texts, s0_images, parity = s0_and_parity(texts, images)
    # Swift's vector for one image moves away from s0 because its decode differs; the FP reference on Swift's pixels
    # moved the same way, so against it Swift is exact.
    moved = images[3]["id"]
    parity["images"][f"{moved}_280"]["emb"] = unit(123).tolist()
    sw_texts, sw_images = tl.swift_vectors(parity, s0_texts, s0_images)
    fp = {70: {}, 280: {moved: unit(123)}}
    refs = {b: tl.image_reference(v["order"], v["ref_bucket"], fp[b]) for b, v in sw_images.items()}
    gated, d = tl.task_level(sw_texts, sw_images, refs, {70: [], 280: [moved]})
    assert set(gated) == set(tl.GATED)
    assert gated["min_cosine"] == pytest.approx(1.0) and gated["min_recall_at_10"] == pytest.approx(1.0)
    assert gated["max_i2i_label_p10_drop"] == pytest.approx(d["vs_fp_on_swift_pixels"]["max_i2i_label_p10_drop"])
    s0_cos = float(np.dot(unit(123), s0_images[280]["emb_bucket"][3]))
    assert d["vs_s0"]["min_cosine"] == pytest.approx(s0_cos, abs=1e-6)
    assert d["vs_s0"]["worst"]["280"][moved] == pytest.approx(s0_cos, abs=1e-6)
    assert d["decoder"]["min_cosine_image"] == pytest.approx(s0_cos, abs=1e-6)  # FP itself moved by the decoder
    assert d["fp_reference"]["280"] == {"fp32_rewrite": 1, "s0_emb_bucket": len(images) - 1}


def test_swift_vectors_follow_s0_order_and_parity_coverage():
    texts, images = fixture_ids(5, 4)
    s0_texts, s0_images, parity = s0_and_parity(texts, images)
    del parity["texts"][texts[2]["id"]], parity["images"][f"{images[0]['id']}_70"]
    t, im = tl.swift_vectors(parity, s0_texts, s0_images)
    assert t["order"] == [x["id"] for n, x in enumerate(texts) if n != 2]
    assert im[70]["order"] == [r["id"] for r in images[1:]] and im[280]["order"] == [r["id"] for r in images]
    np.testing.assert_allclose(im[70]["cos"], 1.0, atol=1e-6)


def test_swift_pixels_checks_presence_shape_and_sha(tmp_path):
    u8 = np.random.default_rng(0).integers(0, 256, (6, 5, 3), dtype=np.uint8)
    Image.fromarray(u8, "RGB").save(tmp_path / "i0_70.png")
    entry = {"height": 6, "width": 5, "sha256": sha(u8)}
    np.testing.assert_array_equal(pixels.swift_pixels(entry, tmp_path, "i0_70"), u8)
    x = pixels.model_input(u8)
    assert x.shape == (1, 3, 6, 5) and x.dtype == np.float32 and x.max() <= 1.0
    with pytest.raises(pixels.PixelDumpError, match="sha256"):
        pixels.swift_pixels(entry | {"sha256": "0" * 64}, tmp_path, "i0_70")
    with pytest.raises(pixels.PixelDumpError, match="shape"):
        pixels.swift_pixels(entry | {"height": 7}, tmp_path, "i0_70")
    with pytest.raises(pixels.PixelDumpError, match="--pixels"):
        pixels.swift_pixels(entry, tmp_path, "i1_70")
    assert pixels.bucket_for("wide", 280).function_name == "embed_image_280_wide"
    with pytest.raises(pixels.PixelDumpError):
        pixels.bucket_for("round", 70)


def test_fp_work_list_is_the_images_whose_pixels_differ_from_s0(tmp_path):
    same, other = np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8)
    np.savez(tmp_path / "images_70.npz", pixels_a=same, pixels_b=same)
    parity = {"images": {"a_70": {"budget": 70, "sha256": sha(same)}, "b_70": {"budget": 70, "sha256": sha(other)},
                         "a_280": {"budget": 280, "sha256": sha(other)}}}
    assert fp_reference.needed(parity, tmp_path, 70) == ["b"]
    refs = {70: {"a": {"u8": same}, "b": {"u8": same}}, 280: {"a": {"u8": same}}}
    assert run.fp_needed(parity, refs) == {70: ["b"], 280: ["a"]}


def test_measure_refuses_a_small_limit_with_the_fp_reference(tmp_path):
    (tmp_path / "m.mlpackage").mkdir()
    (tmp_path / "m.mlpackage" / "Manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="--skip-fp"):
        run.measure(tmp_path / "m.mlpackage", tmp_path, limit=10)


def test_runtime_jobs_cover_everything_grouped_by_function():
    parity = {"texts": {"t0": {"ids": [2, 5, 1]}, "t1": {"ids": list(range(20))}},
              "images": {"i0_70": {"budget": 70, "bucket": "wide"}, "i0_280": {"budget": 280, "bucket": "tall"}}}
    jobs = runtime.jobs(parity, pad_id=0)
    assert [(fn, key) for fn, key, _ in jobs] == [
        ("embed_image_280_tall", "image_i0_280"), ("embed_image_70_wide", "image_i0_70"),
        ("embed_text_16", "text_t0"), ("embed_text_32", "text_t1")]
    text_feed = jobs[2][2]
    assert text_feed["input_ids"].shape == (1, 16) and text_feed["attention_mask"].sum() == 3
