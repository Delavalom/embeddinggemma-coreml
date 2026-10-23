"""s1 parity runs: every fixture text and every image at both budgets through the rewrite, against s0's vectors."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from egcm.buckets import BUCKETS, BUDGETS, Bucket
from egcm.config import TextCfg
from egcm.fixtures import load_bucket_pixels
from egcm.functions import image_embedder
from egcm.metrics import cosine_rows
from egcm.model.embedder import TEXT_LENGTHS, TextEmbedder
from egcm.model.text import TextTower
from egcm.model.vision import VisionEncoder
from stages.s0_reference.pixels import patch_positions

WORST = 3  # lowest-cosine fixtures listed per group in the report


def text_length(n: int) -> int:
    """The smallest text function that fits n tokens."""
    for L in TEXT_LENGTHS:
        if n <= L:
            return L
    raise ValueError(f"{n} tokens is longer than the longest text function ({TEXT_LENGTHS[-1]})")


def _summary(ids: list[str], cos: np.ndarray) -> dict:
    order = np.argsort(cos, kind="stable")[:WORST]
    return {"n": len(ids), "min": float(cos.min()), "mean": float(cos.mean()),
            "worst": [[ids[i], float(cos[i])] for i in order]}


@torch.inference_mode()
def run_texts(tower: TextTower, table: torch.Tensor, cfg: TextCfg, s0: Path) -> dict:
    with np.load(s0 / "texts.npz") as z:
        order = [str(x) for x in z["order"]]
        want = z["emb"]
        ids = {k: z[f"ids_{k}"] for k in order}
    embedders: dict[int, TextEmbedder] = {}
    got, length = [], []
    for k in order:
        n = len(ids[k])
        L = text_length(n)
        if L not in embedders:
            embedders[L] = TextEmbedder(tower, table, L).eval()
        padded = torch.full((1, L), cfg.pad_id, dtype=torch.int32)
        padded[0, :n] = torch.from_numpy(ids[k].astype(np.int32))
        mask = (torch.arange(L) < n).float()[None]
        got.append(embedders[L](padded, mask)[0].numpy())
        length.append(L)
    cos = cosine_rows(np.stack(got), want)
    by_length = {str(L): _summary([k for k, x in zip(order, length, strict=True) if x == L],
                                  cos[np.array(length) == L]) for L in sorted(set(length))}
    return {"min": float(cos.min()), "all": _summary(order, cos), "by_length": by_length}


def _check_inputs(z, image_id: str, layout: dict, bucket: Bucket) -> None:
    """s0's recorded token ids and patch positions for this image are the bucket's (what the embedder assumes)."""
    if z[f"input_ids_{image_id}"].tolist() != layout["buckets"][bucket.function_name]["input_ids"]:
        raise AssertionError(f"{image_id}: input_ids differ from the {bucket.function_name} layout")
    if not np.array_equal(z[f"image_position_ids_{image_id}"], patch_positions(bucket.height, bucket.width)):
        raise AssertionError(f"{image_id}: image_position_ids are not the {bucket.function_name} grid")


@torch.inference_mode()
def run_images(encoder: VisionEncoder, tower: TextTower, table: torch.Tensor, layout: dict, s0: Path,
               log: Callable[[str], None]) -> dict:
    """One bucket at a time (one ImageEmbedder alive), every image of that bucket, both budgets."""
    out: dict = {"by_bucket": {}}
    mins = []
    for budget in BUDGETS:
        with np.load(s0 / f"images_{budget}.npz") as z:
            order = [str(x) for x in z["order"]]
            names = [str(x) for x in z["bucket"]]
            want = z["emb_bucket"]
            for bucket in (b for b in BUCKETS if b.budget == budget):
                idx = [i for i, n in enumerate(names) if n == bucket.name]
                if not idx:
                    continue
                emb = image_embedder(encoder, tower, table, layout, bucket)
                got = []
                for i in idx:
                    _check_inputs(z, order[i], layout, bucket)
                    got.append(emb(torch.from_numpy(load_bucket_pixels(budget, order[i], s0)))[0].numpy())
                cos = cosine_rows(np.stack(got), want[idx])
                out["by_bucket"][bucket.function_name] = _summary([order[i] for i in idx], cos)
                mins.append(float(cos.min()))
                log(f"{bucket.function_name}: {len(idx)} images, min cosine {cos.min():.7f}")
                del emb
    out["min"] = min(mins)
    return out
