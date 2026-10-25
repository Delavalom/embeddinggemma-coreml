import numpy as np
import pytest
from PIL import Image

from egcm.buckets import BUDGETS, choose_bucket
from egcm.paths import ROOT
from stages.s5_swift import run
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
