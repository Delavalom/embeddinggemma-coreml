"""Vision tower: the maths of upstream `Gemma4VisionModel` + `embed_vision`, for one fixed image bucket.

Upstream order (vendor/reference/modeling_gemma4.py, modeling_embedding_gemma2.py):
- Patch embedding (`Gemma4VisionPatchEmbedder`): `input_proj(2 * (patches - 0.5)) + table[0][x] + table[1][y]`,
  patches flattened in (py, px, c) order. Here a 16x16 stride-16 conv whose weight is `input_proj.weight` viewed as
  (out, py, px, c) and permuted to (out, c, py, px), times 2, with bias `-input_proj.weight.sum(1)` (folds the
  `2 * (x - 0.5)`; built by egcm.weights). Position embeddings for the bucket's grid are a constant.
- 16 encoder layers (`Gemma4VisionEncoderLayer`), axial 2-D RoPE (x half, then y half), bidirectional, no padding
  inside a bucket (so no mask).
- Pooler: average of each 3x3 patch block, blocks in row-major order (upstream's `kernel_idxs = x//3 + (Wp/3)*(y//3)`)
  then `* sqrt(768)`. `standardize` is off.
- `embed_vision`: weightless RMSNorm, then Linear(768 -> 512). The weightless RMSNorm is scale-invariant (up to eps),
  so the pooler's `* sqrt(768)` is dropped here.

`VisionEncoder` holds the weights (shared by every bucket); `VisionTower` adds one bucket's constants. Every size is
fixed at construction, so the traced graph has no shape-derived ints.
"""

import torch
import torch.nn.functional as F
from torch import nn

from egcm.buckets import Bucket
from egcm.config import VisionCfg
from egcm.model.attention import AttentionCF, conv1x1
from egcm.model.norm import RMSNormCF
from egcm.model.rope import rope_2d
from egcm.model.text import MLP


class VisionLayer(nn.Module):
    def __init__(self, cfg: VisionCfg):
        super().__init__()
        c, e = cfg.hidden, cfg.eps
        self.attn = AttentionCF(c, cfg.heads, cfg.kv_heads, cfg.head_dim, e, rope_halves=2)
        self.mlp = MLP(c, cfg.ffn)
        self.in_norm, self.post_attn = RMSNormCF(c, e), RMSNormCF(c, e)
        self.pre_ff, self.post_ff = RMSNormCF(c, e), RMSNormCF(c, e)

    def forward(self, h, cos, sin, mask):
        h = h + self.post_attn(self.attn(self.in_norm(h), cos, sin, mask))
        return h + self.post_ff(self.mlp(self.pre_ff(h)))


class VisionEncoder(nn.Module):
    """The vision weights: patch conv, position table, encoder layers, and `embed_vision`'s projection."""

    def __init__(self, cfg: VisionCfg, text_hidden: int):
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(3, cfg.hidden, cfg.patch, stride=cfg.patch, bias=True)
        self.register_buffer("pos_table", torch.zeros(2, cfg.pos_size, cfg.hidden))  # [0] by x, [1] by y
        self.layers = nn.ModuleList([VisionLayer(cfg) for _ in range(cfg.layers)])
        self.mm_norm = RMSNormCF(cfg.hidden, cfg.eps, with_scale=False)
        self.mm_proj = conv1x1(cfg.hidden, text_hidden)


class VisionTower(nn.Module):
    """pixels (1, 3, H, W) in [0, 1] for one bucket -> soft tokens (1, text_hidden, 1, tokens) in text space.

    The bucket's position embeddings are read from `encoder.pos_table` here, so build it after loading the weights."""

    def __init__(self, encoder: VisionEncoder, bucket: Bucket):
        super().__init__()
        cfg = encoder.cfg
        if bucket.height % (cfg.patch * cfg.pool) or bucket.width % (cfg.patch * cfg.pool):
            raise ValueError(f"{bucket.height}x{bucket.width} is not a multiple of {cfg.patch * cfg.pool}")
        self.encoder, self.bucket = encoder, bucket
        self.hp, self.wp = bucket.height // cfg.patch, bucket.width // cfg.patch
        self.patches = self.hp * self.wp
        self.tokens = (self.hp // cfg.pool) * (self.wp // cfg.pool)
        if max(self.hp, self.wp) > cfg.pos_size:
            raise ValueError(f"grid {self.hp}x{self.wp} exceeds the position table ({cfg.pos_size})")
        table = encoder.pos_table
        # pos[c, y, x] = table[0][x, c] + table[1][y, c], the same fp32 add as upstream's x_emb + y_emb
        pos = table[1, : self.hp].T[None, :, :, None] + table[0, : self.wp].T[None, :, None, :]
        self.register_buffer("pos", pos.contiguous(), persistent=False)  # (1, C, Hp, Wp)
        cos, sin = rope_2d(self.hp, self.wp, cfg.head_dim, cfg.theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def embed(self, pixels: torch.Tensor) -> torch.Tensor:
        """Patch + position embeddings, (1, C, 1, patches) in row-major patch order."""
        x = self.encoder.patch(pixels) + self.pos
        return x.reshape(1, self.encoder.cfg.hidden, 1, self.patches)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder.layers:
            h = layer(h, self.cos, self.sin, None)
        return h

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        """3x3 block average, (1, C, 1, tokens), without upstream's `* sqrt(C)` (see the module docstring)."""
        c, k = self.encoder.cfg.hidden, self.encoder.cfg.pool
        g = F.avg_pool2d(h.reshape(1, c, self.hp, self.wp), k, k)
        return g.reshape(1, c, 1, self.tokens)

    def project(self, t: torch.Tensor) -> torch.Tensor:
        return self.encoder.mm_proj(self.encoder.mm_norm(t))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.project(self.pool(self.encode(self.embed(pixels))))
