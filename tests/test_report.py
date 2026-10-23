from egcm.report import Report, write_report, read_report, evaluate

def test_roundtrip_and_pass(tmp_path):
    r = Report(stage="s1_rewrite", git_sha="abc", inputs_sha256="x", metrics={"min_cosine_text": 0.99995},
               gate_values={"min_cosine_text": 0.9999})
    r = evaluate(r, {"min_cosine_text": ">="})
    assert r.passed and r.failures == []
    p = write_report(r, tmp_path)
    assert read_report(p).passed

def test_fail_records_reason(tmp_path):
    r = evaluate(Report(stage="s3_coreml_ane", git_sha="a", inputs_sha256="x",
                        metrics={"min_ane_fraction": 0.8}, gate_values={"min_ane_fraction": 0.95}),
                 {"min_ane_fraction": ">="})
    assert not r.passed
    assert "min_ane_fraction" in r.failures[0]

def test_max_gate(tmp_path):
    r = evaluate(Report(stage="s4_compress", git_sha="a", inputs_sha256="x",
                        metrics={"max_download_mb": 310.0}, gate_values={"max_download_mb": 300}),
                 {"max_download_mb": "<="})
    assert not r.passed
