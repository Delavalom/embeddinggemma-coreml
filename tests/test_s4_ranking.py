"""stages/s4_compress helpers: gate checks, zip and checksums (variant metrics: tests/test_ranking.py)."""

import zipfile

from stages.s4_compress import ranking
from stages.s4_compress.run import METRIC_FOR_GATE, passes


def test_passes_checks_every_gate_in_its_direction():
    g = {"min_cosine": 0.99, "min_recall_at_10": 0.95, "min_p5_spearman_top50": 0.9, "min_t2i_top1_in_top10": 0.95,
         "max_i2i_label_p10_drop": 0.03, "max_download_mb": 360}
    m = {"min_cosine": 0.995, "min_recall_at_10": 0.95, "min_p5_spearman_top50": 0.89, "min_t2i_top1_in_top10": 0.96,
         "max_i2i_label_p10_drop": 0.031, "download_mb": 360.0}
    assert passes(m, g) == {"min_cosine": True, "min_recall_at_10": True, "min_p5_spearman_top50": False,
                            "min_t2i_top1_in_top10": True, "max_i2i_label_p10_drop": False, "max_download_mb": True}


def test_passes_covers_the_gates_in_gates_toml():
    from egcm.gates import gates

    g = gates("s4_compress")
    m = {METRIC_FOR_GATE.get(k, k): v for k, v in g.items()}  # every metric exactly at its gate
    assert set(passes(m, g)) == set(g) and all(passes(m, g).values())


def test_zip_and_checksums(tmp_path):
    pkg = tmp_path / "M.mlpackage"
    (pkg / "Data").mkdir(parents=True)
    (pkg / "Data" / "w.bin").write_bytes(b"x" * 1000)
    (pkg / "Manifest.json").write_text("{}")
    z = ranking.zip_package(pkg, tmp_path / "M.zip")
    assert sorted(zipfile.ZipFile(z).namelist()) == ["M.mlpackage/Data/w.bin", "M.mlpackage/Manifest.json"]
    c = ranking.checksums(pkg, z)
    assert set(c["files"]) == {"M.mlpackage/Data/w.bin", "M.mlpackage/Manifest.json"}
    assert c["archive"]["bytes"] == z.stat().st_size and len(c["archive"]["M.zip"]) == 64
