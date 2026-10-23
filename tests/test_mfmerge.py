"""egcm.mfmerge against coremltools' save_multifunction on the tiny model: same interface, same predictions,
weights stored once, at FP32 and FP16, and with compressed (int4 / int8) weights."""

import platform

import numpy as np
import pytest
import torch

ct = pytest.importorskip("coremltools")

import tiny  # noqa: E402

from egcm import mfmerge  # noqa: E402
from egcm.compress import RECIPES  # noqa: E402
from egcm.convert import _merge, compile_package, contract_problems, convert, size_mb  # noqa: E402
from egcm.evaluate import Package, text_feed  # noqa: E402
from egcm.functions import TextFunction  # noqa: E402
from egcm.model.embedder import TextEmbedder  # noqa: E402

NAMES = ("embed_text_16", "embed_text_32")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    sd = tiny.hf_state_dict(*tiny.reference())
    sd["language_model.embed_tokens.weight"] = torch.randn(4000, tiny.TEXT.hidden)
    text, _, table = tiny.rewrite(sd)
    return text, table


@pytest.mark.parametrize("variant", ["fp32", "fp16", "int4b32_table8"])
def test_matches_save_multifunction(model, tmp_path, variant):
    text, table = model
    srcs = []
    for name in NAMES:
        n = int(name.rsplit("_", 1)[1])
        m = convert(name, TextFunction(TextEmbedder(text.tower, table, n)),
                    (torch.zeros(1, n, dtype=torch.int32), torch.ones(1, n, dtype=torch.int32)),
                    ct.precision.FLOAT32 if variant == "fp32" else ct.precision.FLOAT16)
        if variant in RECIPES:
            m = RECIPES[variant][1](m)
        p = tmp_path / f"{name}.mlpackage"
        m.save(str(p))
        srcs.append((p, "main", name))

    ours, theirs = tmp_path / "ours.mlpackage", tmp_path / "theirs.mlpackage"
    stats = mfmerge.merge(srcs[:1], ours, "embed_text_16")  # incrementally, as egcm.parts does
    stats = mfmerge.merge([(ours, "embed_text_16", "embed_text_16"), srcs[1]], ours, "embed_text_16")
    _merge([(p, "main") for p, _, _ in srcs], theirs)
    assert stats["functions"] == list(NAMES)
    assert stats["blobs_written"] < stats["blob_refs"]  # shared weights stored once
    weights = {k: size_mb(p / "Data" / "com.apple.CoreML" / "weights") for k, p in (("ours", ours), ("theirs", theirs))}
    assert weights["ours"] <= 1.05 * weights["theirs"]
    assert [p for p in contract_problems(ours, NAMES) if "[1, 24]" not in p] == []  # tiny out_dim is 24

    if platform.system() != "Darwin":
        return
    a = Package(compile_package(ours, tmp_path / "ours.mlmodelc"), ct.ComputeUnit.CPU_ONLY)
    b = Package(compile_package(theirs, tmp_path / "theirs.mlmodelc"), ct.ComputeUnit.CPU_ONLY)
    for n in (7, 25):
        fn, feed = text_feed(np.arange(1, n + 1, dtype=np.int32) * 37 % 4000, pad_id=0)
        np.testing.assert_array_equal(a.embed(fn, feed), b.embed(fn, feed))
