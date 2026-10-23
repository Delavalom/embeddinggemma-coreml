import base64
import json
import re
from pathlib import Path

import numpy as np

from egcm.paths import HF_REVISION, ROOT
from stages.s5_swift import export_fixtures as ex


def test_sample_indices_are_spread_and_deterministic():
    idx = ex.sample_indices(3 * 336 * 480, 8192)
    assert len(idx) == 8192 and idx[0] == 0 and idx[-1] < 3 * 336 * 480
    assert np.all(np.diff(idx) > 0)
    assert np.array_equal(idx, ex.sample_indices(3 * 336 * 480, 8192))
    # fewer values than samples: every value once
    assert np.array_equal(ex.sample_indices(10, 8192), np.arange(10))


def test_pick_images_covers_every_shape_and_both_formats():
    rows = [{"id": f"i{i:03d}", "file": f"images/x{i}.{'png' if i % 3 == 0 else 'jpg'}",
             "height": 1000, "width": w} for i, w in enumerate([300, 700, 1000, 1500, 2000] * 6)]
    picked = ex.pick_images(rows, per_shape=4)
    assert len(picked) == 20
    shapes = [ex.shape_of(r) for r in picked]
    assert all(shapes.count(s) == 4 for s in ("tall", "portrait", "square", "landscape", "wide"))
    assert {Path(r["file"]).suffix for r in picked} == {".png", ".jpg"}
    assert picked == ex.pick_images(rows, per_shape=4)


def test_image_entry_samples_chw_uint8():
    u8 = np.arange(48 * 96 * 3, dtype=np.int64).reshape(48, 96, 3).astype(np.uint8)
    e = ex.image_entry(u8, "square", samples=64)
    chw = u8.transpose(2, 0, 1).reshape(-1)
    got = np.frombuffer(base64.b64decode(e["sample_b64"]), np.uint8)
    assert np.array_equal(got, chw[ex.sample_indices(chw.size, 64)])
    assert (e["bucket"], e["height"], e["width"]) == ("square", 48, 96)


def test_swift_tests_pin_the_same_hf_revision():
    src = (ROOT / "swift/Tests/EmbeddingGemmaTests/TestSupport.swift").read_text()
    assert re.search(r'hfRevision = "([0-9a-f]{40})"', src).group(1) == HF_REVISION


def test_committed_fixtures_are_small_and_current():
    d = ROOT / ex.OUT_DIR
    texts, pixels = json.loads((d / "text_ids.json").read_text()), json.loads((d / "pixels.json").read_text())
    assert texts["hf_revision"] == pixels["hf_revision"] == HF_REVISION
    assert len(texts["texts"]) == 200 and len(pixels["images"]) == 20
    assert sum(p.stat().st_size for p in d.iterdir()) < 2_000_000


def test_synthetic_pattern_is_fixed():
    img = ex.synthetic(40, 100)
    assert img.shape == (40, 100, 3) and img.dtype == np.uint8
    # region 0 (gradient), 1 (hash noise), 2 (checkerboard), spot-checked by hand
    assert img[0, 0].tolist() == [0, 60, 120]
    assert img[0, 37].tolist() == [(73856093 * 37 % 2**32) >> 24, ((73856093 * 37) ^ 83492791) % 2**32 >> 24,
                                   ((73856093 * 37) ^ (2 * 83492791)) % 2**32 >> 24]
    assert img[0, 80].tolist() == [0, 0, 0] and img[0, 85].tolist() == [255, 255, 255]  # x // 5 even, odd
    assert set(ex.SYNTHETIC_SIZES) >= {(960, 672), (300, 400)}  # identity at 280, upscale at both budgets


def test_committed_synthetic_cases_cover_every_bucket():
    pixels = json.loads((ROOT / ex.OUT_DIR / "pixels.json").read_text())
    seen = {(b, e["bucket"]) for s in pixels["synthetic"] for b, e in s["budgets"].items()}
    assert len(seen) == 10


def test_committed_embeddings_are_unit_vectors():
    e = json.loads((ROOT / ex.OUT_DIR / "embeddings.json").read_text())
    assert [t["id"] for t in e["texts"]] == list(ex.EMB_TEXTS)
    vecs = [np.frombuffer(base64.b64decode(t["emb_b64"]), "<f4") for t in e["texts"]]
    vecs += [np.frombuffer(base64.b64decode(v), "<f4") for i in e["images"] for v in i["budgets"].values()]
    assert len(vecs) == len(ex.EMB_TEXTS) + 2 * len(ex.EMB_IMAGES)
    assert all(v.shape == (768,) and abs(np.linalg.norm(v) - 1) < 1e-5 for v in vecs)


def test_large_synthetic_cases_are_big_downscales():
    pixels = json.loads((ROOT / ex.OUT_DIR / "pixels.json").read_text())
    cases = pixels["large_synthetic"]
    assert {(c["height"], c["width"], c["budget"]) for c in cases} >= {(3024, 4032, 70), (6048, 8064, 70)}
    assert all(len(c["sha256"]) == 64 for c in cases)
