import hashlib
import math

import numpy as np
import pytest

from stages.s5_swift import compare as c
from stages.s5_swift.export_fixtures import sample_indices

SAMPLES = 64


def entry(u8_hwc: np.ndarray, bucket: str, emb=None, samples: int = SAMPLES) -> dict:
    """What egcm-parity writes for one image (built from the image it would have computed)."""
    h, w, _ = u8_hwc.shape
    chw = np.ascontiguousarray(u8_hwc.transpose(2, 0, 1)).reshape(-1)
    return {"bucket": bucket, "height": h, "width": w,
            "sha256": hashlib.sha256(np.ascontiguousarray(u8_hwc).tobytes()).hexdigest(),
            "pixel_sample": chw[sample_indices(chw.size, samples)].tolist(),
            "emb": None if emb is None else list(map(float, emb))}


def img(seed: int, h: int = 12, w: int = 16) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def unit(seed: int, d: int = 8) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=d)
    return v / np.linalg.norm(v)


def test_id_mismatches_exact_and_both_directions():
    ref = {"a": [2, 5, 1], "b": [2, 7, 1]}
    assert c.id_mismatches({"a": [2, 5, 1], "b": [2, 7, 1]}, ref) == []
    assert c.id_mismatches({"a": [2, 5, 1], "b": [2, 7, 7, 1]}, ref) == ["b"]
    assert c.id_mismatches({"a": [2, 5, 1]}, ref) == ["b"]
    assert c.id_mismatches({**ref, "z": [1]}, ref) == ["z"]


def test_reference_sample_matches_the_cli_rule():
    u8 = img(0)
    chw = u8.transpose(2, 0, 1).reshape(-1)
    s = c.reference_sample(u8, 100)
    assert s.tolist() == [chw[(k * chw.size) // 100] for k in range(100)]
    assert c.reference_sample(u8, 10_000).tolist() == chw.tolist()  # small images: every value


def test_pixel_diffs_in_unit_scale_and_length_check():
    d = c.pixel_diffs([0, 255, 10], np.array([1, 0, 10], np.uint8))
    assert d.dtype == np.float32
    np.testing.assert_allclose(d, [1 / 255, 1.0, 0.0], rtol=1e-6)
    with pytest.raises(ValueError):
        c.pixel_diffs([1, 2], np.array([1, 2, 3], np.uint8))


def test_p99_and_empty_is_inf():
    assert c.p99([np.zeros(99, np.float32), np.ones(1, np.float32)]) == pytest.approx(0.01, abs=1e-6)
    assert c.p99([]) == math.inf


def test_tally_gates_png_and_synthetic_only_and_counts_exact():
    t = c.PixelTally(SAMPLES)
    a, b = img(1), img(2)
    t.add("png", "p", entry(a, "square"), "square", a)
    t.add("synthetic", "s", entry(a, "square"), "square", a)
    t.add("jpeg", "j", entry(b, "square"), "square", a)  # all wrong, but JPEG is not gated
    assert t.metric() == 0.0
    d = t.details()
    assert d["bit_exact"] == {"png": "1/1", "jpeg": "0/1", "synthetic": "1/1"}
    assert d["jpeg_pixel_p99"] > 0.5 and d["png_pixel_p99"] == 0.0


def test_tally_one_off_pixel_is_within_the_gate():
    a = img(3)
    off = a.copy()
    off[0, 0, 0] = (int(a[0, 0, 0]) + 1) % 256
    t = c.PixelTally(10_000)  # every value sampled
    t.add("png", "p", entry(off, "square", samples=10_000), "square", a)
    assert t.metric() <= 0.003922
    assert t.details()["bit_exact"]["png"] == "0/1"


def test_bucket_mismatch_makes_the_metric_inf():
    t = c.PixelTally(SAMPLES)
    a = img(4)
    t.add("png", "p_70", entry(a, "wide"), "square", a)
    assert t.metric() == math.inf
    assert t.details()["bucket_mismatches"] == ["p_70: swift wide 12x16, reference square 12x16"]


def fake_run(text_emb=None, image_emb=None, pixels=None):
    """A complete parity file + references: 2 texts, 2 images (one PNG, one JPEG) at both budgets, 1 synthetic."""
    ref_texts = {"t0": {"ids": [2, 9, 1], "emb": unit(10), "kind": "query"},
                 "t1": {"ids": [2, 8, 8, 1], "emb": unit(11), "kind": "document"}}
    ref_images = {b: {i: {"bucket": "square", "u8": img(20 + n + b), "emb": unit(30 + n + b)}
                      for n, i in enumerate(("i0", "i1"))} for b in (70, 280)}
    ref_synth = {"syn_70": {"bucket": "square", "u8": img(40)}}
    parity = {"samples": SAMPLES, "model_revision": "r", "compute_units": "cpuOnly",
              "texts": {k: {"ids": v["ids"], "emb": list((text_emb or {}).get(k, v["emb"]))}
                        for k, v in ref_texts.items()},
              "images": {f"{i}_{b}": entry((pixels or {}).get(f"{i}_{b}", r["u8"]), r["bucket"],
                                           (image_emb or {}).get(f"{i}_{b}", r["emb"]))
                         for b, refs in ref_images.items() for i, r in refs.items()},
              "synthetic": {k: entry(v["u8"], v["bucket"]) for k, v in ref_synth.items()}}
    return parity, ref_texts, ref_images, {"i0": "png", "i1": "jpeg"}, ref_synth


def test_summarize_identical_run_passes_every_gate():
    m, d = c.summarize(*fake_run())
    assert m["token_ids_identical"] is True
    assert m["max_pixel_abs_diff"] == 0.0
    assert m["min_cosine"] == pytest.approx(1.0)
    assert set(d["cosine"]["text"]) == {"query", "document"}
    assert set(d["cosine"]["image"]) == {"70_png", "70_jpeg", "280_png", "280_jpeg"}
    assert d["missing"] == [] and d["unexpected"] == []


def test_summarize_min_cosine_is_over_texts_and_images_combined():
    m, d = c.summarize(*fake_run(image_emb={"i1_280": unit(99)}))
    expected = float(np.dot(unit(99), unit(30 + 1 + 280)))
    assert m["min_cosine"] == pytest.approx(expected)
    assert d["cosine"]["image"]["280_jpeg"]["worst"] == {"i1": pytest.approx(expected)}
    m, _ = c.summarize(*fake_run(text_emb={"t1": -unit(11)}))
    assert m["min_cosine"] == pytest.approx(-1.0)


def test_summarize_jpeg_pixels_do_not_gate_png_pixels_do():
    m, d = c.summarize(*fake_run(pixels={"i1_70": img(77)}))
    assert m["max_pixel_abs_diff"] == 0.0 and d["pixels"]["jpeg_pixel_p99"] > 0
    m, _ = c.summarize(*fake_run(pixels={"i0_70": img(77)}))
    assert m["max_pixel_abs_diff"] > 0.003922


def test_summarize_missing_outputs_fail_visibly():
    parity, *rest = fake_run()
    del parity["images"]["i0_280"]
    del parity["synthetic"]["syn_70"]
    m, d = c.summarize(parity, *rest)
    assert d["missing"] == ["i0_280", "syn_70"]
    assert m["max_pixel_abs_diff"] == math.inf and math.isnan(m["min_cosine"])

    parity, *rest = fake_run()
    parity["texts"]["t1"]["ids"] = [2, 1]
    m, d = c.summarize(parity, *rest)
    assert m["token_ids_identical"] is False and d["token_id_mismatches"] == ["t1"]
