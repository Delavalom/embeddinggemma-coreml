"""The s0 artifact contract downstream stages rely on. Skipped until `just stage 0` has run."""

import json

import numpy as np
import pytest

from egcm.buckets import BUCKETS, BUDGETS, choose_bucket
from egcm.fixtures import S0_DIR, load_bucket_pixels, load_images, load_texts

pytestmark = pytest.mark.skipif(not (S0_DIR / "images_280.npz").is_file(), reason="run `just stage 0` first")


def test_texts_npz():
    texts = load_texts()
    with np.load(S0_DIR / "texts.npz") as z:
        assert z["order"].tolist() == [t["id"] for t in texts]
        assert z["emb"].shape == (len(texts), 768) and z["emb"].dtype == np.float32
        assert np.allclose(np.linalg.norm(z["emb"], axis=1), 1, atol=1e-5)
        assert all(z[f"ids_{t['id']}"].dtype == np.int32 for t in texts)


@pytest.mark.parametrize("budget", BUDGETS)
def test_images_npz(budget):
    rows = load_images()
    with np.load(S0_DIR / f"images_{budget}.npz") as z:
        assert z["order"].tolist() == [r["id"] for r in rows]
        assert z["bucket"].tolist() == [choose_bucket(r["height"], r["width"], budget).name for r in rows]
        for k in ("emb_raw", "emb_bucket"):
            assert z[k].shape == (len(rows), 768) and np.allclose(np.linalg.norm(z[k], axis=1), 1, atol=1e-5)
        for r, name in zip(rows[:: 17], z["bucket"][:: 17], strict=True):
            b = next(b for b in BUCKETS if b.budget == budget and b.name == name)
            u8 = z[f"pixels_{r['id']}"]
            assert u8.dtype == np.uint8 and u8.shape == (b.height, b.width, 3)
            x = load_bucket_pixels(budget, r["id"])
            assert x.shape == (1, 3, b.height, b.width)
            assert np.array_equal(np.rint(x[0].transpose(1, 2, 0) * 255).astype(np.uint8), u8)  # uint8-exact
            assert len(z[f"image_position_ids_{r['id']}"]) == b.tokens * 9


def test_layout():
    layout = json.loads((S0_DIR / "image_token_layout.json").read_text())
    assert layout["matches_bos_boi_image_eoi_eos"]
    assert set(layout["buckets"]) == {b.function_name for b in BUCKETS}
