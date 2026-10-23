"""stages/s4_compress helpers: variant metrics on synthetic vectors, gate checks, zip and checksums."""

import zipfile

import numpy as np

from egcm.fixtures import load_texts
from stages.s4_compress import ranking
from stages.s4_compress.run import passes


def fake(noise_raw: float, noise_new: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    order = [t["id"] for t in load_texts()]
    q = rng.normal(size=(len(order), 16))
    texts = {"order": order, "ref": q, "emb": q + noise_new * rng.normal(size=q.shape), "cos": np.ones(len(order))}
    images = {}
    for b in (70, 280):
        d = rng.normal(size=(200, 16))
        images[b] = {"ref_bucket": d, "ref_raw": d + noise_raw * rng.normal(size=d.shape),
                     "emb": d + noise_new * rng.normal(size=d.shape), "cos": np.ones(200)}
    return texts, images


def test_identical_vectors_rank_identically():
    m = ranking.metrics(*fake(0.0, 0.0))
    assert m["min_recall_at_10"] == 1.0 and m["min_p5_spearman_top50"] == 1.0
    assert m["vs_bucket"]["min_recall_at_10"] == 1.0 and m["min_cosine"] == 1.0


def test_raw_reference_includes_bucketing_cost():
    m = ranking.metrics(*fake(noise_raw=0.5, noise_new=0.0))
    assert m["vs_bucket"]["min_recall_at_10"] == 1.0
    assert m["min_recall_at_10"] < 0.95  # gated metric is against the exact-aspect reference


def test_passes_uses_each_gate_direction():
    g = {"min_cosine": 0.99, "min_recall_at_10": 0.95, "min_p5_spearman_top50": 0.9, "max_download_mb": 300}
    m = {"min_cosine": 0.995, "min_recall_at_10": 0.95, "min_p5_spearman_top50": 0.89, "download_mb": 301}
    assert passes(m, g) == {"min_cosine": True, "min_recall_at_10": True, "min_p5_spearman_top50": False,
                            "max_download_mb": False}


def test_zip_and_checksums(tmp_path):
    pkg = tmp_path / "M.mlpackage"
    (pkg / "Data").mkdir(parents=True)
    (pkg / "Data" / "w.bin").write_bytes(b"x" * 1000)
    (pkg / "Manifest.json").write_text("{}")
    z = ranking.zip_package(pkg, tmp_path / "M.zip")
    assert sorted(zipfile.ZipFile(z).namelist()) == ["M.mlpackage/Data/w.bin", "M.mlpackage/Manifest.json"]
    c = ranking.checksums(pkg, z)
    assert set(c["files"]) == {"M.mlpackage/Data/w.bin", "M.mlpackage/Manifest.json"}
    assert c["zip"]["bytes"] == z.stat().st_size and len(c["zip"]["M.zip"]) == 64
