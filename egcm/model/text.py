"""Text tower: the maths of upstream `EmbeddingGemma2TextModel` in the Neural Engine layout (1, C, 1, S).

Upstream order (vendor/reference/modeling_embedding_gemma2.py):
- PLE (`EmbeddingGemma2TextPLE`): `RMSNorm_P(reshape(Linear(C -> L*P)(inputs_embeds) * C**-0.5, [S, L, P]))`,
  projection-only (no token lookup). The reshape is layer-major, so layer i's slice is channels i*P .. (i+1)*P of
  the projection: `split(P, dim=1)`.
- Layer i (`EmbeddingGemma2EncoderLayer`):
    h = h + post_attention_layernorm(attn(input_layernorm(h)))
    h = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
    h = h + post_per_layer_input_norm(per_layer_projection(gelu_tanh(per_layer_input_gate(h)) * ple_i))
    h = h * layer_scalar
- Output: `embedding_projection(norm(h))`, per token (pooling is the embedder's job).

The tower holds the weights only. RoPE tables depend on the sequence length, so the caller (an embedder built for one
fixed length) passes them in: one tower is shared by every text length and image bucket.
"""

import torch
from torch import nn

from egcm.config import TextCfg
from egcm.model.attention import AttentionCF, conv1x1
from egcm.model.norm import RMSNormCF
from egcm.model.rope import rope_1d

Rope = dict[str, tuple[torch.Tensor, torch.Tensor]]  # layer type -> (cos, signed_sin), each (1, head_dim, 1, S)


class MLP(nn.Module):
    """Gated GELU-tanh MLP (`gelu_pytorch_tanh` in both towers' configs)."""

    def __init__(self, c: int, f: int):
        super().__init__()
        self.gate, self.up, self.down = conv1x1(c, f), conv1x1(c, f), conv1x1(f, c)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


class TextLayer(nn.Module):
    def __init__(self, cfg: TextCfg, kind: str):
        super().__init__()
        c, e = cfg.hidden, cfg.eps
        self.kind = kind
        self.attn = AttentionCF(c, cfg.heads, cfg.kv_heads[kind], cfg.head_dim[kind], e, rope_halves=1)
        self.mlp = MLP(c, cfg.ffn)
        self.in_norm, self.post_attn = RMSNormCF(c, e), RMSNormCF(c, e)
        self.pre_ff, self.post_ff = RMSNormCF(c, e), RMSNormCF(c, e)
        self.ple_gate, self.ple_proj = conv1x1(c, cfg.ple_dim), conv1x1(cfg.ple_dim, c)
        self.post_ple = RMSNormCF(c, e)
        self.act = nn.GELU(approximate="tanh")
        self.register_buffer("layer_scalar", torch.ones(1))  # persistent upstream: loaded from the weights

    def forward(self, h, ple_i, rope: Rope, mask):
        cos, sin = rope[self.kind]
        h = h + self.post_attn(self.attn(self.in_norm(h), cos, sin, mask))
        h = h + self.post_ff(self.mlp(self.pre_ff(h)))
        h = h + self.post_ple(self.ple_proj(self.act(self.ple_gate(h)) * ple_i))
        return h * self.layer_scalar


class TextTower(nn.Module):
    """(inputs_embeds, mask, rope) -> per-token (1, out_dim, 1, S).

    inputs_embeds (1, C, 1, S) is already scaled (text tokens by sqrt(C); image soft tokens unscaled); mask is the
    additive key-major (1, S, 1, 1) from `key_mask`, or None when nothing is padded; rope is `rope_tables(cfg, S)`."""

    def __init__(self, cfg: TextCfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([TextLayer(cfg, kind) for kind in cfg.layer_types])
        self.ple_proj = conv1x1(cfg.hidden, cfg.layers * cfg.ple_dim)
        self.ple_norm = RMSNormCF(cfg.ple_dim, cfg.eps)
        self.ple_scale = cfg.hidden**-0.5
        self.norm = RMSNormCF(cfg.hidden, cfg.eps)
        self.out_proj = conv1x1(cfg.hidden, cfg.out_dim)

    def per_layer_inputs(self, emb: torch.Tensor) -> list[torch.Tensor]:
        """L tensors (1, P, 1, S): upstream `per_layer_inputs[:, :, i, :]` in channels-first layout."""
        ple = self.ple_proj(emb) * self.ple_scale  # (1, L*P, 1, S), layer-major along channels
        return [self.ple_norm(p) for p in ple.split(self.cfg.ple_dim, dim=1)]

    def forward(self, emb: torch.Tensor, mask: torch.Tensor | None, rope: Rope) -> torch.Tensor:
        h = emb
        for layer, ple_i in zip(self.layers, self.per_layer_inputs(emb), strict=True):
            h = layer(h, ple_i, rope, mask)
        return self.out_proj(self.norm(h))


def rope_tables(cfg: TextCfg, seq_len: int) -> Rope:
    """Text RoPE (cos, signed_sin) for positions 0..seq_len-1, one pair per layer type in the config."""
    return {kind: rope_1d(seq_len, cfg.head_dim[kind], cfg.theta[kind]) for kind in sorted(set(cfg.layer_types))}

