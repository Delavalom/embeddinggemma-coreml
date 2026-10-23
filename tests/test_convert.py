"""egcm.convert on the tiny model: contract I/O, batched multifunction merge with deduplicated weights, and
prediction through a compiled package with egcm.evaluate."""

import platform

import numpy as np
import pytest
import torch

ct = pytest.importorskip("coremltools")

import tiny  # noqa: E402

from egcm.convert import build_package, compile_package, contract_problems, convert, size_mb  # noqa: E402
from egcm.evaluate import Package, text_feed  # noqa: E402
from egcm.functions import TextFunction  # noqa: E402
from egcm.model.embedder import TextEmbedder  # noqa: E402

NAMES = ("embed_text_16", "embed_text_32", "embed_text_64")


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pkg")
    torch.manual_seed(0)
    sd = tiny.hf_state_dict(*tiny.reference())
    sd["language_model.embed_tokens.weight"] = torch.randn(4000, tiny.TEXT.hidden)  # big enough to see dedup
    text, _, table = tiny.rewrite(sd)

    def make(name):
        n = int(name.rsplit("_", 1)[1])
        m = TextFunction(TextEmbedder(text.tower, table, n))
        return convert(name, m, (torch.zeros(1, n, dtype=torch.int32), torch.ones(1, n, dtype=torch.int32)),
                       ct.precision.FLOAT32)

    out = tmp / "M.mlpackage"
    sizes = build_package(NAMES, make, out, tmp / "parts", batch=2, log=lambda _: None)
    return out, sizes, text, tmp


def test_merge_keeps_every_function_and_the_contract(package):
    out, _, _, tmp = package
    assert not (tmp / "parts").exists()
    problems = contract_problems(out, NAMES)
    # the tiny model's out_dim is 24, not 768; everything else must match
    assert all("outputs" in p and "[1, 24]" in p for p in problems), problems
    assert len(problems) == len(NAMES)


def test_weights_are_stored_once(package):
    out, sizes, _, _ = package
    weights = size_mb(out / "Data" / "com.apple.CoreML" / "weights")
    table_mb = 4000 * tiny.TEXT.hidden * 4 / 1e6
    assert table_mb < weights < 1.5 * table_mb  # one copy of the table (plus the small tower), not three
    assert size_mb(out) < sum(sizes["parts_mb"].values())


@pytest.mark.skipif(platform.system() != "Darwin", reason="Core ML runtime needs macOS")
def test_compiled_functions_match_torch(package):
    out, _, text, tmp = package
    pkg = Package(compile_package(out, tmp / "M.mlmodelc"), ct.ComputeUnit.CPU_ONLY)
    for n in (5, 20, 40):
        ids = np.random.default_rng(n).integers(1, 4000, n).astype(np.int32)
        fn, feed = text_feed(ids, pad_id=0)
        L = int(fn.rsplit("_", 1)[1])
        want = TextEmbedder(text.tower, text.table, L)(torch.from_numpy(feed["input_ids"]),
                                                      torch.from_numpy(feed["attention_mask"]).float())
        got = pkg.embed(fn, feed)
        assert float(got @ want[0].detach().numpy()) > 1 - 1e-5
