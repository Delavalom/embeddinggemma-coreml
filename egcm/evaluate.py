"""Run every s0 fixture through a compiled multifunction package (s2-s4) and collect the embeddings.

Inputs are exactly the contract's: texts padded with `pad_id` to the smallest text function that fits, an int32
attention mask; images from `egcm.fixtures.load_bucket_pixels` for the bucket s0 chose.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import coremltools as ct
import numpy as np

from egcm.buckets import BUCKETS, BUDGETS
from egcm.fixtures import S0_DIR, load_bucket_pixels
from egcm.metrics import cosine_rows
from egcm.model.embedder import TEXT_LENGTHS

WORST = 3


def text_length_for(n: int) -> int:
    for L in TEXT_LENGTHS:
        if n <= L:
            return L
    raise ValueError(f"{n} tokens is longer than the longest text function ({TEXT_LENGTHS[-1]})")


class Package:
    """Loads one function of a compiled package at a time (each load is a full model load)."""

    def __init__(self, compiled: Path, compute_units: ct.ComputeUnit):
        self.compiled, self.units = Path(compiled), compute_units
        self._name: str | None = None
        self._model = None

    def function(self, name: str):
        if name != self._name:
            self._model = None
            self._model = ct.models.CompiledMLModel(str(self.compiled), compute_units=self.units, function_name=name)
            self._name = name
        return self._model

    def embed(self, name: str, feed: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(self.function(name).predict(feed)["embedding"], dtype=np.float32).reshape(-1)


def text_feed(ids: np.ndarray, pad_id: int) -> tuple[str, dict[str, np.ndarray]]:
    L = text_length_for(len(ids))
    padded = np.full((1, L), pad_id, dtype=np.int32)
    padded[0, : len(ids)] = ids
    mask = (np.arange(L) < len(ids)).astype(np.int32)[None]
    return f"embed_text_{L}", {"input_ids": padded, "attention_mask": mask}


def summary(ids: list[str], cos: np.ndarray) -> dict:
    order = np.argsort(cos, kind="stable")[:WORST]
    return {"n": len(ids), "min": float(cos.min()), "mean": float(cos.mean()),
            "worst": [[ids[i], float(cos[i])] for i in order]}


def run_texts(pkg: Package, pad_id: int, s0: Path = S0_DIR) -> dict:
    """{"order", "kind" (from texts.npz order), "emb" [N, 768], "ref" [N, 768], "by_function"}."""
    with np.load(s0 / "texts.npz") as z:
        order = [str(x) for x in z["order"]]
        ref = z["emb"]
        ids = {k: z[f"ids_{k}"] for k in order}
    feeds = {k: text_feed(ids[k], pad_id) for k in order}
    got = np.zeros_like(ref)
    for fn in sorted({f for f, _ in feeds.values()}, key=lambda f: int(f.rsplit("_", 1)[1])):
        for i, k in enumerate(order):
            if feeds[k][0] == fn:
                got[i] = pkg.embed(fn, feeds[k][1])
    cos = cosine_rows(got, ref)
    fns = [feeds[k][0] for k in order]
    by_fn = {fn: summary([k for k, f in zip(order, fns, strict=True) if f == fn], cos[np.array(fns) == fn])
             for fn in sorted(set(fns))}
    return {"order": order, "emb": got, "ref": ref, "cos": cos, "by_function": by_fn}


def run_images(pkg: Package, s0: Path = S0_DIR, log: Callable[[str], None] = print) -> dict:
    """Per budget: {"order", "emb", "ref_bucket", "ref_raw", "cos"}; plus "by_function" over both budgets."""
    out: dict = {"by_function": {}}
    for budget in BUDGETS:
        with np.load(s0 / f"images_{budget}.npz") as z:
            order = [str(x) for x in z["order"]]
            names = [str(x) for x in z["bucket"]]
            ref, raw = z["emb_bucket"], z["emb_raw"]
        got = np.zeros_like(ref)
        for b in (b for b in BUCKETS if b.budget == budget):
            idx = [i for i, n in enumerate(names) if n == b.name]
            if not idx:
                continue
            t = time.monotonic()
            for i in idx:
                got[i] = pkg.embed(b.function_name, {"pixels": load_bucket_pixels(budget, order[i], s0)})
            cos = cosine_rows(got[idx], ref[idx])
            out["by_function"][b.function_name] = summary([order[i] for i in idx], cos)
            log(f"{b.function_name}: {len(idx)} images, min cosine {cos.min():.6f} "
                f"({(time.monotonic() - t) / len(idx):.2f} s/image incl. load)")
        out[budget] = {"order": order, "emb": got, "ref_bucket": ref, "ref_raw": raw, "cos": cosine_rows(got, ref)}
    return out


def latency_ms(pkg: Package, name: str, feed: dict[str, np.ndarray], runs: int = 20) -> dict:
    """Median and p90 wall time of `predict` after 3 warm-up calls (the model is loaded first)."""
    m = pkg.function(name)
    for _ in range(3):
        m.predict(feed)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        m.predict(feed)
        ts.append((time.perf_counter() - t) * 1e3)
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)), "runs": runs}
