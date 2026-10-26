from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from egcm.buckets import BUDGETS, choose_bucket
from egcm.paths import ROOT
from stages.s5_swift import models, run, runtime
from stages.s5_swift.export_fixtures import SYNTHETIC_SIZES, synthetic


def test_runs_in_the_convert_group():
    assert (ROOT / "stages/s5_swift/run.py").read_text().splitlines()[0] == "# group: convert"


def test_measure_only_flags_are_refused_for_the_stage():
    with pytest.raises(SystemExit, match="only apply with --measure"):
        run.main(["--limit", "3"])


def test_synthetic_pngs_are_lossless_and_refs_cover_both_budgets(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "SYNTHETIC_SIZES", SYNTHETIC_SIZES[:2])
    monkeypatch.setattr(run, "torch_resize", lambda u8, h, w: np.zeros((h, w, 3), np.uint8))  # torch tested elsewhere
    refs = run.write_synthetic(tmp_path)
    for h, w in SYNTHETIC_SIZES[:2]:
        with Image.open(tmp_path / f"syn_{h}x{w}.png") as im:
            assert im.mode == "RGB" and np.array_equal(np.asarray(im), synthetic(h, w))
        for b in BUDGETS:
            bk = choose_bucket(h, w, b)
            ref = refs[f"syn_{h}x{w}_{b}"]
            assert ref["bucket"] == bk.name and ref["u8"].shape == (bk.height, bk.width, 3)
    assert len(refs) == 2 * len(BUDGETS)


def test_measure_needs_a_package(tmp_path):
    with pytest.raises(FileNotFoundError, match="just stage 4"):
        run.measure(tmp_path / "missing.mlpackage", tmp_path)
    (tmp_path / "b70.mlpackage").mkdir()
    (tmp_path / "b70.mlpackage" / "Manifest.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match=r"b280\.mlpackage"):  # every budget's package (ADR-0010)
        run.measure({70: tmp_path / "b70.mlpackage", 280: tmp_path / "b280.mlpackage"}, tmp_path)


def test_model_arguments_per_budget_or_one_package():
    two = {280: Path("/m/b280.mlpackage"), 70: Path("/m/b70.mlpackage")}
    assert models.model_args(two) == ["--model", "70=/m/b70.mlpackage", "--model", "280=/m/b280.mlpackage"]
    assert models.model_args(Path("/m/all.mlpackage")) == ["--model", "/m/all.mlpackage"]
    assert models.parse_models(["70=/m/b70.mlpackage", "280=/m/b280.mlpackage"]) == two
    assert models.parse_models(["/m/all.mlpackage"]) == {70: Path("/m/all.mlpackage"), 280: Path("/m/all.mlpackage")}
    for bad in (["70=/a"], ["70=/a", "70=/b"], ["140=/a", "280=/b"], ["/a", "/b"]):
        with pytest.raises(ValueError, match="--model"):
            models.parse_models(bad)


def test_runtime_runs_each_budgets_images_on_its_package_and_text_on_one():
    assert (models.budget_of("embed_image_280_tall"), models.budget_of("embed_image_70_wide")) == (280, 70)
    assert models.budget_of("embed_text_128") == models.TEXT_BUDGET == 70
    todo = [("embed_image_280_tall", "image_a_280", "a_280"), ("embed_image_70_wide", "image_a_70", "a_70"),
            ("embed_text_16", "text_t0", {})]
    two = {70: Path("b70"), 280: Path("b280")}
    assert {p: [k for _, k, _ in js] for p, js in runtime.by_package(two, todo).items()} == {
        Path("b280"): ["image_a_280"], Path("b70"): ["image_a_70", "text_t0"]}
    one = models.parse_models(["all"])
    assert list(runtime.by_package(one, todo)) == [Path("all")]  # one compile for a 14-function package
