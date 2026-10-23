"""google/embeddinggemma-2 safetensors -> the rewrite's modules, strict.

Every Hugging Face key the rewrite uses maps to exactly one destination by one rule (`RULES`); linear weights become
1x1 conv weights `(out, in) -> (out, in, 1, 1)`, and the patch projection a 16x16 conv weight. The modules are
created on the meta device and filled with `load_state_dict(strict=True, assign=True)`, so a missing or extra key
raises and no weight is held twice. Keys `keys.json` lists that no rule uses must all belong to the audio tower
(tests/test_weights_map.py).

Load-time check: with `scaling = 1.0`, q and k leave their RMSNorms with rms 1 per head before the weights, so
`|q.k| <= head_dim * max|q_norm.w| * max|k_norm.w|`. `check_logit_bound` rejects a checkpoint above `LOGIT_BOUND`,
which could push logits past what the FP16 softmax and the -1e4 mask margin are designed for.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open

from egcm.buckets import Bucket
from egcm.config import TextCfg, VisionCfg
from egcm.model.attention import AttentionCF
from egcm.model.text import TextTower
from egcm.model.vision import VisionEncoder, VisionTower

LOGIT_BOUND = 64.0
TABLE_KEY = "language_model.embed_tokens.weight"

# (tower, HF key pattern, destination template, transform). "conv": (out, in) -> (out, in, 1, 1); "copy": as is;
# "patch": the patch projection as a stride-16 conv weight (see _patch).
_N = r"(?P<i>\d+)"
RULES: tuple[tuple[str, str, str, str], ...] = (
    ("table", re.escape(TABLE_KEY), "", "copy"),
    ("text", r"language_model\.ple\.per_layer_model_projection\.weight", "ple_proj.weight", "conv"),
    ("text", r"language_model\.ple\.per_layer_projection_norm\.weight", "ple_norm.weight", "copy"),
    ("text", rf"language_model\.layers\.{_N}\.self_attn\.(?P<p>[qkvo])_proj\.weight", "layers.{i}.attn.{p}.weight",
     "conv"),
    ("text", rf"language_model\.layers\.{_N}\.self_attn\.(?P<p>[qk])_norm\.weight", "layers.{i}.attn.{p}_norm.weight",
     "copy"),
    ("text", rf"language_model\.layers\.{_N}\.input_layernorm\.weight", "layers.{i}.in_norm.weight", "copy"),
    ("text", rf"language_model\.layers\.{_N}\.post_attention_layernorm\.weight", "layers.{i}.post_attn.weight",
     "copy"),
    ("text", rf"language_model\.layers\.{_N}\.pre_feedforward_layernorm\.weight", "layers.{i}.pre_ff.weight", "copy"),
    ("text", rf"language_model\.layers\.{_N}\.post_feedforward_layernorm\.weight", "layers.{i}.post_ff.weight",
     "copy"),
    ("text", rf"language_model\.layers\.{_N}\.mlp\.(?P<p>gate|up|down)_proj\.weight", "layers.{i}.mlp.{p}.weight",
     "conv"),
    ("text", rf"language_model\.layers\.{_N}\.ple_block\.per_layer_input_gate\.weight", "layers.{i}.ple_gate.weight",
     "conv"),
    ("text", rf"language_model\.layers\.{_N}\.ple_block\.per_layer_projection\.weight", "layers.{i}.ple_proj.weight",
     "conv"),
    ("text", rf"language_model\.layers\.{_N}\.ple_block\.post_per_layer_input_norm\.weight",
     "layers.{i}.post_ple.weight", "copy"),
    ("text", rf"language_model\.layers\.{_N}\.layer_scalar", "layers.{i}.layer_scalar", "copy"),
    ("text", r"language_model\.norm\.weight", "norm.weight", "copy"),
    ("text", r"language_model\.embedding_projection\.weight", "out_proj.weight", "conv"),
    ("vision", r"vision_tower\.patch_embedder\.input_proj\.weight", "patch.weight", "patch"),
    ("vision", r"vision_tower\.patch_embedder\.position_embedding_table", "pos_table", "copy"),
    ("vision", rf"vision_tower\.encoder\.layers\.{_N}\.self_attn\.(?P<p>[qkvo])_proj\.linear\.weight",
     "layers.{i}.attn.{p}.weight", "conv"),
    ("vision", rf"vision_tower\.encoder\.layers\.{_N}\.self_attn\.(?P<p>[qk])_norm\.weight",
     "layers.{i}.attn.{p}_norm.weight", "copy"),
    ("vision", rf"vision_tower\.encoder\.layers\.{_N}\.input_layernorm\.weight", "layers.{i}.in_norm.weight", "copy"),
    ("vision", rf"vision_tower\.encoder\.layers\.{_N}\.post_attention_layernorm\.weight",
     "layers.{i}.post_attn.weight", "copy"),
    ("vision", rf"vision_tower\.encoder\.layers\.{_N}\.pre_feedforward_layernorm\.weight", "layers.{i}.pre_ff.weight",
     "copy"),
    ("vision", rf"vision_tower\.encoder\.layers\.{_N}\.post_feedforward_layernorm\.weight",
     "layers.{i}.post_ff.weight", "copy"),
    ("vision", rf"vision_tower\.encoder\.layers\.{_N}\.mlp\.(?P<p>gate|up|down)_proj\.linear\.weight",
     "layers.{i}.mlp.{p}.weight", "conv"),
    ("vision", r"embed_vision\.embedding_projection\.weight", "mm_proj.weight", "conv"),
)
_COMPILED = tuple((tower, re.compile(pat), dest, how) for tower, pat, dest, how in RULES)


def _strip(key: str) -> str:
    return key.removeprefix("model.")


def destination(key: str) -> tuple[str, str, str] | None:
    """(tower, destination name, transform) for a HF key, or None if the rewrite does not use it."""
    k = _strip(key)
    for tower, pat, dest, how in _COMPILED:
        if m := pat.fullmatch(k):
            return tower, dest.format(**m.groupdict()), how
    return None


def mapping_for_keys(keys: Iterable[str]) -> tuple[list[str], list[str]]:
    """(used, unused) HF keys, each sorted."""
    used, unused = [], []
    for k in sorted(keys):
        (used if destination(k) is not None else unused).append(k)
    return used, unused


def load_hf(snapshot: Path) -> dict[str, torch.Tensor]:
    """The used tensors of `<snapshot>/model.safetensors` as FP32 (audio and other unused keys are not read)."""
    out = {}
    with safe_open(Path(snapshot) / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            if destination(k) is not None:
                out[_strip(k)] = f.get_tensor(k).float()
    return out


def _patch(w: torch.Tensor, patch: int) -> torch.Tensor:
    """input_proj (C, p*p*3), input index (py, px, c) -> conv weight (C, 3, p, p). The model applies 2 * (x - 0.5)."""
    return w.view(w.shape[0], patch, patch, 3).permute(0, 3, 1, 2).contiguous()


def tower_state_dict(sd: dict[str, torch.Tensor], tower: str, patch: int = 16) -> dict[str, torch.Tensor]:
    """The state dict of `TextTower` ("text") or `VisionEncoder` ("vision") from HF tensors."""
    out: dict[str, torch.Tensor] = {}
    for k, t in sd.items():
        d = destination(k)
        if d is None or d[0] != tower:
            continue
        _, name, how = d
        out[name] = _patch(t, patch) if how == "patch" else t[:, :, None, None] if how == "conv" else t
    return out


def check_logit_bound(module: torch.nn.Module, bound: float = LOGIT_BOUND) -> dict[str, float]:
    """`head_dim * max|q_norm.w| * max|k_norm.w|` per attention module; raises ValueError if any exceeds `bound`."""
    found = {}
    for name, m in module.named_modules():
        if isinstance(m, AttentionCF):
            found[name] = float(m.head_dim * m.q_norm.weight.abs().max() * m.k_norm.weight.abs().max())
    if bad := {k: v for k, v in found.items() if v > bound}:
        raise ValueError(f"q.k logit bound above {bound} (FP16 softmax / mask margin): {bad}")
    return found


def _load(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> torch.nn.Module:
    module.load_state_dict(state, strict=True, assign=True)  # raises on a missing or unexpected key
    module.requires_grad_(False)
    check_logit_bound(module)
    return module.eval()


def build_text_tower(cfg: TextCfg, sd: dict[str, torch.Tensor]) -> TextTower:
    with torch.device("meta"):
        tower = TextTower(cfg)
    return _load(tower, tower_state_dict(sd, "text"))


def build_vision_encoder(vcfg: VisionCfg, sd: dict[str, torch.Tensor], text_hidden: int) -> VisionEncoder:
    with torch.device("meta"):
        enc = VisionEncoder(vcfg, text_hidden)
    return _load(enc, tower_state_dict(sd, "vision", vcfg.patch))


def build_vision_tower(encoder: VisionEncoder, bucket: Bucket) -> VisionTower:
    return VisionTower(encoder, bucket).eval()


def text_table(sd: dict[str, torch.Tensor], hidden: int) -> torch.Tensor:
    """`embed_tokens.weight * sqrt(hidden)`: upstream `EmbeddingGemma2TextScaledWordEmbedding` in FP32."""
    return sd[TABLE_KEY] * torch.tensor(hidden**0.5, dtype=torch.float32)
