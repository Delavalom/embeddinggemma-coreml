"""egcm.published.shipped: s5, s6 and s7 take the package and recipe from s4's report, never from a hard-coded name."""

import pytest

from egcm import published
from egcm.report import Report, write_report
from stages.s5_swift import run as s5
from stages.s6_device import run as s6
from stages.s7_publish import run as s7


def put_s4(root, **details):
    d = root / "artifacts" / "s4_compress"
    d.mkdir(parents=True, exist_ok=True)
    winner = details.get("winner")
    write_report(Report(stage="s4_compress", status="passed" if winner else "failed", passed=bool(winner),
                        metrics={"max_download_mb": 350.2}, gate_values={"max_download_mb": 360},
                        details={"package": "EmbeddingGemma2.mlpackage", **details}), d)
    return d


def test_shipped_reads_the_report(tmp_path):
    d = put_s4(tmp_path, chosen="some_recipe", winner="some_recipe")
    s4 = published.shipped(tmp_path)
    assert s4 == published.Shipped(package=d / "EmbeddingGemma2.mlpackage", recipe="some_recipe", download_mb=350.2,
                                   max_download_mb=360)


def test_shipped_refuses_no_report_and_no_winner(tmp_path):
    with pytest.raises(FileNotFoundError, match="just stage 4"):
        published.shipped(tmp_path)
    put_s4(tmp_path, chosen="closest", winner=None)  # a failed s4 still leaves its closest variant in place
    with pytest.raises(RuntimeError, match="shipped nothing"):
        published.shipped(tmp_path)


def shipped(tmp_path, recipe="int8_table4"):
    return published.Shipped(package=tmp_path / "s4" / "EmbeddingGemma2.mlpackage", recipe=recipe, download_mb=350.2,
                             max_download_mb=360)


def test_s5_measures_the_shipped_package_and_records_its_recipe(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(s5.published, "shipped", lambda: shipped(tmp_path, "r1"))
    monkeypatch.setattr(s5, "stage_dir", lambda _: tmp_path / "out")
    monkeypatch.setattr(s5, "measure", lambda model, out: (seen.setdefault("model", model) and {"m": 1}, {"d": 2}))
    r = Report(stage="s5_swift")
    s5.fill(r)
    assert seen["model"] == tmp_path / "s4" / "EmbeddingGemma2.mlpackage"
    assert r.metrics == {"m": 1} and r.details == {"d": 2, "s4_recipe": "r1"}


def test_s6_defaults_to_the_shipped_package(tmp_path, monkeypatch):
    class Bundled(Exception):
        pass

    def assemble(package, tokenizer, out):
        raise Bundled(package)

    monkeypatch.setattr(s6.published, "shipped", lambda: shipped(tmp_path))
    monkeypatch.setattr(s6.published, "tokenizer_source", lambda: tmp_path)
    monkeypatch.setattr(s6.published, "assemble", assemble)
    monkeypatch.setattr(s6, "stage_dir", lambda _: tmp_path / "out")
    with pytest.raises(Bundled) as e:
        s6.make_fill(s6.argparse.Namespace(package=None, device=None))(Report(stage="s6_device"))
    assert e.value.args[0] == tmp_path / "s4" / "EmbeddingGemma2.mlpackage"


def test_s7_preflight_names_the_shipped_recipe(tmp_path):
    pub = s7.Publisher(run=lambda cmd: s7.subprocess.CompletedProcess(cmd, 1, "", ""))
    put_s4(tmp_path, chosen="int8_table4", winner="int8_table4")
    problems = s7.preflight(tmp_path, "v1.0.0", pub)
    assert any("s4 recipe int8_table4" in p and "is missing" in p for p in problems)

    put_s4(tmp_path, chosen="closest", winner=None)
    problems = s7.preflight(tmp_path, "v1.0.0", pub)
    assert any(p.startswith("s4_compress: s4 shipped nothing") for p in problems)


def test_model_card_names_the_recipe_and_download(tmp_path):
    card = s7.model_card("me/model", "| r |", shipped(tmp_path))
    assert "s4 recipe `int8_table4`, 350 MB zipped" in card and "| r |" in card
    assert "recipe" not in s7.model_card("me/model", "| r |")
