from egcm.report import Report, write_report
from scripts.gen_results import render


def put(root, stage, **kw):
    d = root / "artifacts" / stage
    d.mkdir(parents=True, exist_ok=True)
    write_report(Report(stage=stage, **kw), d)


def test_results_with_no_reports(repo):
    text = render(repo)
    assert "| s0_reference | not run |" in text and "## s4 variants" not in text


def test_gate_rows_say_which_gate_failed(repo):
    put(repo, "s1_rewrite", status="failed", metrics={"min_a": 0.4, "min_b": 0.9},
        gate_values={"min_a": 0.5, "min_b": 0.5}, failures=["min_a: 0.4 not >= 0.5"])
    put(repo, "s0_reference", status="error", gate_values={"min_a": 0.5}, failures=["RuntimeError: boom"])
    text = render(repo)
    assert "| s1_rewrite | failed (stale) | min_a | 0.4 | >= 0.5 | no |" in text
    assert "| s1_rewrite | failed (stale) | min_b | 0.9 | >= 0.5 | yes |" in text
    assert "| s0_reference | error (stale) | min_a | - | >= 0.5 | - |" in text  # never evaluated


def test_every_s4_variant_and_s5_pixels_are_listed(repo):
    v = {"download_mb": 350.2, "package_mb": 360.0, "min_cosine": 0.991, "min_recall_at_10": 0.96,
         "min_p5_spearman_top50": 0.93, "min_t2i_top1_in_top10": 0.97, "max_i2i_label_p10_drop": 0.01,
         "ane_fraction_min": 0.99, "vs_raw": {"min_recall_at_10": 0.85}, "description": "int8, table int4"}
    put(repo, "s4_compress", status="passed", passed=True, gate_values={"max_download_mb": 360}, details={
        "chosen": "int8_table4", "winner": "int8_table4",
        "variants": {"int8_table4": {**v, "passes": {"min_cosine": True}},
                     "int8": {**v, "download_mb": 410.0, "passes": {"min_cosine": True, "max_download_mb": False}}}})
    put(repo, "s5_swift", status="passed", passed=True, details={
        "pixels": {"png_pixel_p99": 0.0, "png_pixel_max": 0.0, "jpeg_pixel_p99": 0.0078, "jpeg_pixel_max": 0.12,
                   "synthetic_pixel_p99": 0.0, "synthetic_pixel_max": 0.0,
                   "bit_exact": {"png": "60/60", "jpeg": "0/340", "synthetic": "20/20"}},
        "cosine": {"text": {"query": {"n": 120, "min": 0.995, "mean": 0.999}}, "image": {}}})
    text = render(repo)
    assert "Chosen: `int8_table4` (winner: `int8_table4`)" in text and "Download limit: 360 MB." in text
    row = "| int8_table4 | 350.2 | 360 | 0.991 | 0.96 | 0.93 | 0.97 | 0.01 | 0.99 | 0.85 | all | int8, table int4 |"
    assert row in text
    assert "| int8 | 410 |" in text and "fails max_download_mb" in text
    assert "| png | 0 | 0 | 60/60 |" in text and "| jpeg | 0.0078 | 0.12 | 0/340 |" in text
    assert "| text query | 120 | 0.995 | 0.999 |" in text


def test_device_results_from_docs_results(repo):
    d = repo / "docs" / "results"
    d.mkdir(parents=True)
    write_report(Report(stage="s6_device", status="passed", passed=True, metrics={"max_peak_rss_mb": 500.0},
                        gate_values={"max_peak_rss_mb": 800}, details={"device": "iPhone16,1"}), d)
    (d / "report.json").rename(d / "s6_iPhone16,1.json")
    assert "| docs/results/s6_iPhone16,1.json | iPhone16,1 | passed | max_peak_rss_mb=500 |" in render(repo)


def test_s6_device_comes_from_its_devices(repo):
    put(repo, "s6_device", status="passed", passed=True, metrics={"max_peak_rss_mb": 500.0},
        gate_values={"max_peak_rss_mb": 800}, details={"devices": {"iPhone16,1": {}, "iPhone14,3": {}}})
    assert "| artifacts/s6_device/report.json | iPhone16,1, iPhone14,3 | passed |" in render(repo)
