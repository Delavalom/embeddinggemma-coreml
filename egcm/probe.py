"""Activation max-abs probe of the FP32 rewrite (diagnostic only; s3 records it, nothing is gated on it).

Records max |x| per site over the fixtures it is given:
- the residual stream after every layer (`TextLayer` / `VisionLayer` outputs), the embeddings entering the text tower
  (`TextTower` input) and the patch embedding;
- every attention and MLP output, the MLP's inner product (input of `mlp.down`), the PLE gate product (input of
  `ple_proj`);
- every RMSNormCF input (no flag needed: the dynamic norm scales each token by its own max, so any input is
  FP16-safe) and output.

Anything above `FP16_RISK` (FP16 max is 65504) is flagged `overflow_risk`.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager

import torch
from torch import nn

from egcm.model.attention import AttentionCF
from egcm.model.norm import RMSNormCF
from egcm.model.text import MLP, TextLayer, TextTower
from egcm.model.vision import VisionLayer

FP16_RISK = 30000.0


def _kind(m: nn.Module) -> str | None:
    if isinstance(m, TextLayer | VisionLayer):
        return "residual"
    if isinstance(m, AttentionCF):
        return "attn_out"
    if isinstance(m, MLP):
        return "mlp_out"
    if isinstance(m, RMSNormCF):
        return "norm_out"
    return None


@contextmanager
def probe(root: nn.Module, prefix: str = ""):
    """Context manager yielding {prefix + module path + ":" + kind: max |x|}, updated by every forward of `root`
    inside the block. Use prefixes like "text/" and "image/" to keep the text tower's sites apart by input."""
    store: dict[str, float] = {}

    def record(name: str, t: torch.Tensor) -> None:
        v = float(t.detach().abs().max())
        store[name] = max(store.get(name, 0.0), v)

    hooks = []
    for name, m in root.named_modules():
        full = f"{prefix}{name}"
        kind = _kind(m)
        if kind:
            hooks.append(m.register_forward_hook(lambda _m, _i, out, n=full, k=kind: record(f"{n}:{k}", out)))
        if isinstance(m, RMSNormCF):
            hooks.append(m.register_forward_pre_hook(lambda _m, args, n=full: record(f"{n}:norm_in", args[0])))
        if isinstance(m, MLP):
            hooks.append(m.down.register_forward_pre_hook(lambda _m, args, n=full: record(f"{n}:mlp_inner", args[0])))
        if isinstance(m, TextLayer):
            hooks.append(m.ple_proj.register_forward_pre_hook(
                lambda _m, args, n=full: record(f"{n}:ple_inner", args[0])))
        if name.endswith("encoder.patch"):
            hooks.append(m.register_forward_hook(lambda _m, _i, out, n=full: record(f"{n}:patch_out", out)))
        if isinstance(m, TextTower):
            hooks.append(m.register_forward_pre_hook(lambda _m, args, n=full: record(f"{n}:embeddings", args[0])))
    try:
        yield store
    finally:
        for h in hooks:
            h.remove()


def report(stores: Iterable[dict[str, float]], top: int = 25) -> dict:
    """Merge stores (max per site) -> {"max_by_kind", "top" (sorted descending), "overflow_risk"}."""
    merged: dict[str, float] = {}
    for s in stores:
        for k, v in s.items():
            merged[k] = max(merged.get(k, 0.0), v)
    ranked = sorted(merged.items(), key=lambda kv: -kv[1])
    by_kind: dict[str, float] = {}
    for k, v in merged.items():
        kind = k.rsplit(":", 1)[1]
        side = k.split("/", 1)[0] if "/" in k else ""
        tower = "vision" if "vision." in k else "text"
        key = f"{side}/{tower}:{kind}" if side else f"{tower}:{kind}"
        by_kind[key] = max(by_kind.get(key, 0.0), v)
    return {"sites": len(merged), "max_by_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
            "top": ranked[:top], "overflow_risk": [[k, v] for k, v in ranked if v > FP16_RISK],
            "all": dict(ranked)}
