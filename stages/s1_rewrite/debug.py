"""s1 diagnostics: the rewrite's intermediates against s0's debug.npz (t000; i000 at budget 70). Not gated.

Each entry is {max_abs, ref_max_abs, rel = max_abs / ref_max_abs}, in pipeline order, so the first tensor with a
large `rel` is where a parity failure starts. Upstream layouts are (1, S, C); ours are (1, C, 1, S).
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from egcm.buckets import BUCKETS
from egcm.fixtures import to_unit
from egcm.model.text import TextTower, rope_tables
from egcm.model.vision import VisionEncoder
from egcm.weights import build_vision_tower
from stages.s0_reference.pixels import patch_positions
from stages.s1_rewrite.parity import embed_ids, image_affixes

TEXT_LAYERS = (0, 5, 23)
VISION_LAYERS = (0, 15)


def tokens(x: torch.Tensor) -> np.ndarray:
    """(1, C, 1, S) -> (S, C) numpy."""
    return x[0, :, 0].T.numpy()


def diff(got: np.ndarray, want: np.ndarray) -> dict[str, float]:
    if got.shape != want.shape:
        raise AssertionError(f"shape {got.shape} != reference {want.shape}")
    err, scale = float(np.abs(got - want).max()), float(np.abs(want).max())
    return {"max_abs": err, "ref_max_abs": scale, "rel": err / scale if scale else err}


@contextmanager
def outputs(modules: dict[str, torch.nn.Module]):
    store: dict[str, torch.Tensor] = {}
    hooks = [m.register_forward_hook(lambda _m, _i, out, k=k: store.__setitem__(k, out.detach()))
             for k, m in modules.items()]
    try:
        yield store
    finally:
        for h in hooks:
            h.remove()


def _text_side(tower: TextTower, emb: torch.Tensor, dbg, prefix: str) -> dict[str, dict]:
    """inputs_embeds -> PLE -> layers -> final norm -> per-token projection, against `<prefix>_lm_*`."""
    res = {"lm_inputs_embeds": diff(tokens(emb), dbg[f"{prefix}_lm_inputs_embeds"][0])}
    ple = np.stack([tokens(p) for p in tower.per_layer_inputs(emb)], 1)  # (S, L, P)
    res["lm_per_layer_inputs"] = diff(ple, dbg[f"{prefix}_lm_per_layer_inputs"][0])
    mods = {f"lm_layer_{i:02d}": tower.layers[i] for i in TEXT_LAYERS} | {"lm_final_norm": tower.norm}
    with outputs(mods) as got:
        last = tower(emb, None, rope_tables(tower.cfg, emb.shape[-1]))
    for k in mods:
        res[k] = diff(tokens(got[k]), dbg[f"{prefix}_{k}"][0])
    res["lm_last_hidden_state"] = diff(tokens(last), dbg[f"{prefix}_lm_last_hidden_state"][0])
    return res


@torch.inference_mode()
def compare_text(tower: TextTower, table: torch.Tensor, dbg) -> dict[str, dict]:
    return _text_side(tower, embed_ids(table, dbg["text_input_ids"].tolist()), dbg, "text")


@torch.inference_mode()
def compare_image(encoder: VisionEncoder, tower: TextTower, table: torch.Tensor, layout: dict, dbg) -> dict:
    u8 = dbg["image_pixels"]
    h, w = u8.shape[:2]
    bucket = next(b for b in BUCKETS if b.budget == 70 and (b.height, b.width) == (h, w))
    if not np.array_equal(dbg["image_position_ids"], patch_positions(h, w)):
        raise AssertionError("debug image positions are not the bucket grid")
    if dbg["image_input_ids"].tolist() != layout["buckets"][bucket.function_name]["input_ids"]:
        raise AssertionError(f"debug image input_ids differ from the {bucket.function_name} layout")
    vt = build_vision_tower(encoder, bucket)
    pixels = torch.from_numpy(np.ascontiguousarray(to_unit(u8).transpose(2, 0, 1)[None]))
    n, t = vt.patches, vt.tokens  # upstream pads the patch rows to budget * 9 and strips them after pooling

    res = {"bucket": bucket.function_name}
    x = vt.embed(pixels)
    res["vision_patch_embed"] = diff(tokens(x), dbg["image_vision_patch_embed"][0, :n])
    mods = {f"vision_layer_{i:02d}": encoder.layers[i] for i in VISION_LAYERS}
    with outputs(mods) as got:
        h = vt.encode(x)
    for k in mods:
        res[k] = diff(tokens(got[k]), dbg[f"image_{k}"][0, :n])
    pooled = vt.pool(h)
    # the rewrite drops upstream's * sqrt(C) (scale-invariant before embed_vision); put it back to compare
    res["vision_pooler"] = diff(tokens(pooled) * np.float32(encoder.cfg.hidden**0.5), dbg["image_vision_pooler"][0, :t])
    soft = vt.project(pooled)
    res["embed_vision"] = diff(tokens(soft), dbg["image_embed_vision"][:t])
    pre, suf = image_affixes(layout, bucket)
    emb = torch.cat([embed_ids(table, pre), soft, embed_ids(table, suf)], dim=-1)
    return res | _text_side(tower, emb, dbg, "image")
