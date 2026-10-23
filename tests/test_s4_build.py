"""stages/s4_compress/build.py: one FP16 conversion per function feeds one package per recipe, weights stored once."""

import pytest
import torch

ct = pytest.importorskip("coremltools")

import tiny  # noqa: E402

from egcm.compress import RECIPES  # noqa: E402
from egcm.convert import contract_problems, convert, size_mb  # noqa: E402
from egcm.functions import TextFunction  # noqa: E402
from egcm.model.embedder import TextEmbedder  # noqa: E402
from stages.s4_compress.build import build_variants  # noqa: E402

NAMES = ("embed_text_16", "embed_text_32", "embed_text_64")


def test_one_package_per_recipe_with_shared_weights(tmp_path):
    torch.manual_seed(0)
    sd = tiny.hf_state_dict(*tiny.reference())
    sd["language_model.embed_tokens.weight"] = torch.randn(8000, tiny.TEXT.hidden)
    text, _, table = tiny.rewrite(sd)
    calls = []

    def make(name):
        calls.append(name)
        n = int(name.rsplit("_", 1)[1])
        m = TextFunction(TextEmbedder(text.tower, table, n))
        return convert(name, m, (torch.zeros(1, n, dtype=torch.int32), torch.ones(1, n, dtype=torch.int32)),
                       ct.precision.FLOAT16)

    recipes = {r: f for r, (_, f) in RECIPES.items()}
    sizes = build_variants(NAMES, make, recipes, tmp_path, log=lambda _: None)
    assert calls == list(NAMES)  # converted once per function, not once per recipe
    assert not (tmp_path / "parts").exists()
    weights = {r: size_mb(tmp_path / f"{r}.mlpackage" / "Data" / "com.apple.CoreML" / "weights") for r in recipes}
    table_int8_mb = 8000 * tiny.TEXT.hidden / 1e6
    assert weights["int8"] < 1.6 * table_int8_mb  # one copy of the int8 table, not three
    assert weights["int4b32"] < 0.75 * weights["int8"]
    for r in recipes:
        assert set(sizes[r]["parts_mb"]) == set(NAMES)
        assert all("outputs" in p for p in contract_problems(tmp_path / f"{r}.mlpackage", NAMES))  # out_dim 24
