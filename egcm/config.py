"""Typed model config read from the Hugging Face `config.json` of google/embeddinggemma-2.

`load` also checks the assumptions the rewrite is built on (rope types, no standardize, no clipped linears, no
attention bias, one head_dim / kv_heads per layer type, a sliding window covering every sequence we build) and
raises ValueError if the file breaks one of them.
"""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from egcm.buckets import MAX_SEQ
from egcm.paths import ARTIFACTS, HF_REPO, HF_REVISION

# Where s0's snapshot_download(cache_dir=artifacts/hf) puts HF_REVISION (config.json, safetensors, tokenizer).
# Kept here, not in paths.py, so rewrite-only additions don't make s0 stale.
HF_SNAPSHOT = ARTIFACTS / "hf" / f"models--{HF_REPO.replace('/', '--')}" / "snapshots" / HF_REVISION

LAYER_TYPES = ("sliding_attention", "full_attention")


class FrozenMap(Mapping):
    """A read-only, hashable mapping, so the frozen config dataclasses keep value equality and can be hashed
    (a dict field would make `hash(cfg)` raise). Compares equal to a dict with the same items."""

    def __init__(self, items: Mapping):
        self._d = dict(items)

    def __getitem__(self, key):
        return self._d[key]

    def __iter__(self) -> Iterator:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __hash__(self) -> int:
        return hash(frozenset(self._d.items()))

    def __repr__(self) -> str:
        return f"FrozenMap({self._d!r})"


@dataclass(frozen=True)
class TextCfg:
    hidden: int
    layers: int
    heads: int
    ffn: int
    ple_dim: int
    out_dim: int
    vocab: int
    eps: float
    layer_types: tuple[str, ...]
    head_dim: FrozenMap  # layer type -> int
    kv_heads: FrozenMap  # layer type -> int
    theta: FrozenMap  # layer type -> float
    pad_id: int
    sliding_window: int


@dataclass(frozen=True)
class VisionCfg:
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    patch: int
    pool: int
    theta: float
    eps: float
    pos_size: int


def _per_type(t: dict, field: str, default_key: str) -> FrozenMap:
    """Resolve a per-layer field to one value per layer type.

    `per_layer_config` holds overrides keyed by 2-digit layer index ("05"); layers without an entry use the
    top-level value. Every layer of one type must end up with the same value.
    """
    overrides = t.get("per_layer_config") or {}
    unknown = set(overrides) - {f"{i:02d}" for i in range(t["num_hidden_layers"])}
    if unknown:
        raise ValueError(f"per_layer_config has keys that are not layer indices: {sorted(unknown)}")
    seen: dict[str, tuple[int, str]] = {}
    for i, kind in enumerate(t["layer_types"]):
        key = f"{i:02d}"
        value = overrides.get(key, {}).get(field, t[default_key])
        if kind not in seen:
            seen[kind] = (value, key)
        elif seen[kind][0] != value:
            raise ValueError(f"layer {key} ({kind}) has {field}={value}, layer {seen[kind][1]} has {seen[kind][0]}")
    return FrozenMap({kind: value for kind, (value, _) in seen.items()})


def _no_attention_bias(c: dict, tower: str) -> None:
    if c.get("attention_bias", False):
        raise ValueError(f"{tower} attention_bias is true; the rewrite's q/k/v/o convs have no bias")


def _text(t: dict) -> TextCfg:
    _no_attention_bias(t, "text")
    if t["sliding_window"] < MAX_SEQ:
        raise ValueError(
            f"sliding_window {t['sliding_window']} < MAX_SEQ {MAX_SEQ}: sliding layers would need their own mask"
        )
    layer_types = tuple(t["layer_types"])
    if len(layer_types) != t["num_hidden_layers"] or set(layer_types) - set(LAYER_TYPES):
        raise ValueError(f"unexpected layer_types {layer_types}")
    rope = t["rope_parameters"]
    for kind in set(layer_types):
        if rope[kind]["rope_type"] != "default":
            raise ValueError(f"text rope_type for {kind} is {rope[kind]['rope_type']}; only 'default' is implemented")
    return TextCfg(
        hidden=t["hidden_size"], layers=t["num_hidden_layers"], heads=t["num_attention_heads"],
        ffn=t["intermediate_size"], ple_dim=t["hidden_size_per_layer_input"], out_dim=t["embedding_dim"],
        vocab=t["vocab_size"], eps=t["rms_norm_eps"], layer_types=layer_types,
        head_dim=_per_type(t, "head_dim", "head_dim"),
        kv_heads=_per_type(t, "num_key_value_heads", "num_key_value_heads"),
        theta=FrozenMap({kind: float(rope[kind]["rope_theta"]) for kind in set(layer_types)}),
        pad_id=t["pad_token_id"], sliding_window=t["sliding_window"],
    )


def _vision(v: dict) -> VisionCfg:
    _no_attention_bias(v, "vision")
    if v["rope_parameters"]["rope_type"] != "axial":
        raise ValueError(f"vision rope_type is {v['rope_parameters']['rope_type']}; only 'axial' is implemented")
    for flag in ("standardize", "use_clipped_linears"):
        if v.get(flag):
            raise ValueError(f"vision {flag} is true; the rewrite assumes it is false")
    return VisionCfg(
        hidden=v["hidden_size"], layers=v["num_hidden_layers"], heads=v["num_attention_heads"],
        kv_heads=v["num_key_value_heads"], head_dim=v["head_dim"], ffn=v["intermediate_size"],
        patch=v["patch_size"], pool=v["pooling_kernel_size"], theta=float(v["rope_parameters"]["rope_theta"]),
        eps=v["rms_norm_eps"], pos_size=v["position_embedding_size"],
    )


def load(hf_dir: Path) -> tuple[TextCfg, VisionCfg, dict]:
    """Parse `<hf_dir>/config.json`. Returns (text, vision, raw dict)."""
    c = json.loads((Path(hf_dir) / "config.json").read_text())
    return _text(c["text_config"]), _vision(c["vision_config"]), c
