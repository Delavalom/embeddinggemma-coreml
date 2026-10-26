"""s4's split into one package per image budget (ADR-0010): the bookkeeping, and the identity check on a tiny real
Core ML package (the full model is never converted here)."""

import json
import platform

import numpy as np
import pytest
import torch

ct = pytest.importorskip("coremltools")

import tiny  # noqa: E402

from egcm import manifest, mfmerge  # noqa: E402
from egcm.buckets import BUCKETS  # noqa: E402
from egcm.convert import convert  # noqa: E402
from egcm.functions import TextFunction  # noqa: E402
from egcm.model.embedder import TextEmbedder  # noqa: E402
from egcm.report import Report  # noqa: E402
from stages.s4_compress import run as s4  # noqa: E402
from stages.s4_compress import split  # noqa: E402

on_mac = pytest.mark.skipif(platform.system() != "Darwin", reason="Core ML predictions need macOS")


def test_names_and_feeds_follow_the_contract():
    assert split.names(70) == ("EmbeddingGemma2-b70.mlpackage", "EmbeddingGemma2-b70.aar")
    for fn in manifest.REQUIRED_FUNCTIONS:
        x = split.feed(fn)
        again = split.feed(fn)
        assert all(np.array_equal(x[k], again[k]) for k in x)  # deterministic
        if fn.startswith("embed_text_"):
            n = int(fn.rsplit("_", 1)[1])
            assert {k: (v.dtype, v.shape) for k, v in x.items()} == {"input_ids": (np.int32, (1, n)),
                                                                      "attention_mask": (np.int32, (1, n))}
            assert x["input_ids"][0, 0] == 2 and 0 < x["attention_mask"].sum() < n
        else:
            b = next(b for b in BUCKETS if b.function_name == fn)
            assert x["pixels"].dtype == np.float32 and x["pixels"].shape == (1, 3, b.height, b.width)


def test_compare_is_bitwise():
    a = np.array([0.5, -0.25, np.nan], np.float32)
    assert split.compare(a, a.copy())["identical"]
    near = a.copy()
    near[0] = np.nextafter(near[0], np.float32(1))
    assert not split.compare(a, near)["identical"]
    assert split.compare(a, a[:2]) == {"identical": False, "max_abs_diff": float("inf")}
    same = {"embed_text_16": {"identical": True}}
    assert split.summarize({"70": same, "280": same})["identical"]
    out = split.summarize({"70": same, "280": {"embed_text_16": {"identical": False}}})
    assert not out["identical"] and out["budgets"]["70"]["identical"] and not out["budgets"]["280"]["identical"]
    assert not split.summarize({"70": {}})["identical"]  # no function compared proves nothing


def test_split_bookkeeping(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(split, "_child", lambda args, tmp: calls.append(args) or {"blobs_written": 9,
                                                                               "peak_footprint_gb": 1.5})
    budget_of = lambda p: int(p.stem.rsplit("-b", 1)[1])  # noqa: E731  (EmbeddingGemma2-b70 -> 70)
    monkeypatch.setattr(split, "function_names", lambda p: sorted(manifest.required_functions(budget_of(p))))
    monkeypatch.setattr("egcm.convert.contract_problems", lambda p, names: [])

    def archive(pkg, tok, out):
        out.write_bytes(b"x" * 2_000_000)
        return out

    monkeypatch.setattr("egcm.archive.device_archive", archive)
    for b in (70, 280):
        (tmp_path / split.names(b)[0]).mkdir()
        (tmp_path / split.names(b)[0] / "w").write_bytes(b"y" * 1_000_000)
    res = split.split(tmp_path / "EmbeddingGemma2.mlpackage", tmp_path, tmp_path / "tok", log=lambda _: None)
    assert [c[0] for c in calls] == ["extract", "extract"] and [c[3] for c in calls] == ["70", "280"]
    assert res["70"] == {"package": "EmbeddingGemma2-b70.mlpackage", "device_archive": "EmbeddingGemma2-b70.aar",
                         "functions": sorted(manifest.required_functions(70)), "package_mb": 1.0,
                         "download_mb": 2.0, "blobs_written": 9, "peak_footprint_gb": 1.5}
    monkeypatch.setattr(split, "function_names", lambda p: sorted(manifest.REQUIRED_FUNCTIONS))
    with pytest.raises(AssertionError, match="holds"):
        split.split(tmp_path / "EmbeddingGemma2.mlpackage", tmp_path, tmp_path / "tok", log=lambda _: None)


def test_ship_budgets_sets_the_download_metric_and_refuses_a_changed_split(tmp_path, monkeypatch):
    final = tmp_path / "EmbeddingGemma2.mlpackage"
    final.mkdir()
    (final / "Manifest.json").write_text("{}")
    budgets = {}
    for b, mb in ((70, 343.1), (280, 352.8)):
        pkg, aar = split.names(b)
        (tmp_path / pkg).mkdir()
        (tmp_path / pkg / "Manifest.json").write_text(str(b))
        (tmp_path / aar).write_bytes(str(b).encode())
        budgets[str(b)] = {"package": pkg, "device_archive": aar, "download_mb": mb, "functions": []}
    same = {"identical": True, "budgets": {}}
    monkeypatch.setattr(split, "split", lambda src, out, tok, log: budgets)
    monkeypatch.setattr(split, "identity", lambda src, packages, work, log: same)
    r = Report(stage="s4_compress", metrics={"max_download_mb": 358.7})
    s4.ship_budgets(r, final, tmp_path, lambda _: None)
    assert r.metrics["max_download_mb"] == 352.8 and r.details["budgets"] == budgets
    sums = json.loads((tmp_path / "checksums.json").read_text())
    assert set(sums["archive"]) == {"EmbeddingGemma2-b70.aar", "EmbeddingGemma2-b280.aar"}
    assert "EmbeddingGemma2-b280.mlpackage/Manifest.json" in sums["files"]

    (tmp_path / "checksums.json").unlink()
    differs = {"identical": False, "budgets": {"280": {"functions": {"embed_text_16": {"identical": False}}}}}
    monkeypatch.setattr(split, "identity", lambda src, packages, work, log: differs)
    with pytest.raises(RuntimeError, match="b280/embed_text_16"):
        s4.ship_budgets(r, final, tmp_path, lambda _: None)
    assert r.details["split_identity"] == differs and not (tmp_path / "checksums.json").exists()


@pytest.fixture(scope="module")
def tiny_packages(tmp_path_factory):
    """A 2-function package (the "14-function" source), and another model's package for one of its functions."""
    d = tmp_path_factory.mktemp("split")
    out = {}
    for seed in (0, 1):
        sd = tiny.hf_state_dict(*tiny.reference(seed))  # reseeds torch: two different models
        sd["language_model.embed_tokens.weight"] = torch.randn(4000, tiny.TEXT.hidden)
        text, _, table = tiny.rewrite(sd)
        srcs = []
        for n in (16, 32):
            m = convert(f"embed_text_{n}", TextFunction(TextEmbedder(text.tower, table, n)),
                        (torch.zeros(1, n, dtype=torch.int32), torch.ones(1, n, dtype=torch.int32)),
                        ct.precision.FLOAT16)
            m.save(str(d / f"{seed}_{n}.mlpackage"))
            srcs.append((d / f"{seed}_{n}.mlpackage", "main", f"embed_text_{n}"))
        out[seed] = d / f"all_{seed}.mlpackage"
        mfmerge.merge(srcs, out[seed], "embed_text_16")
    return out[0], out[1]


@on_mac
def test_identity_on_a_real_split(tiny_packages, tmp_path):
    src, other = tiny_packages
    part = tmp_path / "part.mlpackage"
    mfmerge.merge([(src, "embed_text_32", "embed_text_32")], part, "embed_text_16")  # what `extract` does
    assert split.function_names(part) == ["embed_text_32"]
    wrong = tmp_path / "wrong.mlpackage"
    mfmerge.merge([(other, "embed_text_16", "embed_text_16")], wrong, "embed_text_16")
    res = split.identity(src, {70: part, 280: wrong}, tmp_path / "identity", log=lambda _: None)  # child process
    assert res["budgets"]["70"] == {"identical": True, "functions": {
        "embed_text_32": {"identical": True, "max_abs_diff": 0.0}}}
    assert not res["budgets"]["280"]["identical"] and not res["identical"]
    assert res["budgets"]["280"]["functions"]["embed_text_16"]["max_abs_diff"] > 0
    assert not (tmp_path / "identity").exists()  # the compiled models are deleted
