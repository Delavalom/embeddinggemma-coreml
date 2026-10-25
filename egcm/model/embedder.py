"""The Core ML entry points: token ids or pixels in, one L2-normalized embedding (1, out_dim) out.

Pooling is sentence-transformers' mean over the attention mask (`include_prompt: true`, so the prompt, bos and eos
count), then L2 normalization. Both are written so no intermediate leaves FP16 range: per-token outputs reach about
2,860, so a plain sum over 284 tokens (about 8e5) or a plain sum of squares of the pooled vector (about 6e9) would not
fit in 65,504; on the Neural Engine that measured cosine 0.92 for images (text, with 16 tokens, passed).

Each embedder is built for one fixed sequence length and owns that length's RoPE tables; the towers are shared, so
building many embedders does not copy weights.
"""

import torch
import torch.nn.functional as F
from torch import nn

from egcm.model.masks import key_mask
from egcm.model.text import TextTower, rope_tables
from egcm.model.vision import VisionTower

# Sequence lengths of the text functions (embed_text_<L>); a text uses the smallest that fits.
TEXT_LENGTHS = (16, 32, 64, 128)


# Lower bound for max |v| in l2_normalize (an all-zero vector stays finite, as in RMSNormCF).
MIN_VECTOR_MAX = 2.0**-6


def l2_normalize(v: torch.Tensor) -> torch.Tensor:
    """v / ||v|| over the last dim, FP16-safe: v is divided by its own max |v| first, so the sum of squares is at
    most the dimension (768) instead of overflowing FP16 (pooled values reach about 2,800, squares about 8e6).
    Identical in exact maths."""
    vs = v / v.abs().amax(-1, keepdim=True).clamp_min(MIN_VECTOR_MAX)
    return vs * torch.rsqrt((vs * vs).sum(-1, keepdim=True))


class _RopeOwner(nn.Module):
    def _register_rope(self, tower: TextTower, seq_len: int) -> None:
        self.kinds = sorted(set(tower.cfg.layer_types))
        for kind, (cos, sin) in rope_tables(tower.cfg, seq_len).items():
            self.register_buffer(f"cos_{kind}", cos, persistent=False)
            self.register_buffer(f"sin_{kind}", sin, persistent=False)

    def _rope(self):
        return {k: (getattr(self, f"cos_{k}"), getattr(self, f"sin_{k}")) for k in self.kinds}


class TextEmbedder(_RopeOwner):
    """input_ids (1, S) int32 and attention_mask (1, S) float 0/1 -> (1, out_dim).

    `table` is `embed_tokens.weight * sqrt(hidden)` (egcm.weights.text_table); the lookup happens here."""

    def __init__(self, tower: TextTower, table: torch.Tensor, seq_len: int):
        super().__init__()
        self.tower, self.seq_len = tower, seq_len
        self.register_buffer("table", table, persistent=False)  # (V, C)
        self._register_rope(tower, seq_len)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = F.embedding(input_ids, self.table).transpose(1, 2).unsqueeze(2)  # (1, C, 1, S)
        tokens = self.tower(emb, key_mask(attention_mask), self._rope())  # (1, D, 1, S)
        valid = attention_mask.view(1, 1, 1, -1)
        weights = valid / valid.sum(-1, keepdim=True)  # each token weighted by 1/n before the sum (FP16-safe mean)
        pooled = (tokens * weights).sum(-1, keepdim=True)
        return l2_normalize(pooled.reshape(1, -1))


class ImageEmbedder(_RopeOwner):
    """pixels (1, 3, H, W) in [0, 1] for one bucket -> (1, out_dim).

    The sequence is `prefix + soft tokens + suffix`: the prefix and suffix are the fixed non-image tokens of s0's
    recorded layout (bos, <|image> / <image|>, eos), embedded with the scaled table and baked in as constants, so the
    image functions do not carry the embedding table. Every token is valid, so no mask, and pooling is a plain mean.

    The sequence is not built with a concat along the sequence axis: on the Neural Engine (M1 Pro, macOS 26) that
    returned garbage for the tokens after the soft tokens (the suffix starts at an unaligned offset, 62 for
    embed_image_70_tall; cosine 0.0001 for eos), which took the image cosine to 0.90. Instead
    `emb = affixes + soft @ place`: `affixes` is the (1, C, 1, S) constant with the prefix and suffix embeddings and
    zeros at the soft-token positions, and `place` the (T, S) 0/1 matrix that puts soft token t at position P + t.
    Exact in FP32 and FP16 (every output is one product by 1 plus zeros).
    """

    def __init__(self, vision: VisionTower, tower: TextTower, prefix_emb: torch.Tensor, suffix_emb: torch.Tensor):
        super().__init__()
        self.vision, self.tower = vision, tower
        c, p, t, q = prefix_emb.shape[1], prefix_emb.shape[-1], vision.tokens, suffix_emb.shape[-1]
        self.channels, self.seq_len = c, p + t + q  # fixed at construction
        affixes = torch.zeros(1, c, 1, self.seq_len)
        affixes[..., :p], affixes[..., p + t:] = prefix_emb, suffix_emb
        place = torch.zeros(t, self.seq_len)
        place[torch.arange(t), p + torch.arange(t)] = 1.0
        self.register_buffer("prefix", prefix_emb, persistent=False)  # (1, C, 1, P), kept for inspection
        self.register_buffer("suffix", suffix_emb, persistent=False)  # (1, C, 1, Q)
        self.register_buffer("affixes", affixes, persistent=False)
        self.register_buffer("place", place, persistent=False)
        self._register_rope(tower, self.seq_len)

    def sequence(self, pixels: torch.Tensor) -> torch.Tensor:
        """inputs_embeds (1, C, 1, S): the affixes plus the soft tokens at their positions."""
        soft = self.vision(pixels).reshape(1, self.channels, -1)  # (1, C, T)
        return self.affixes + torch.matmul(soft, self.place).reshape(1, self.channels, 1, self.seq_len)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        tokens = self.tower(self.sequence(pixels), None, self._rope())
        pooled = (tokens * (1.0 / self.seq_len)).sum(-1, keepdim=True)  # mean, scaled before the sum (FP16-safe)
        return l2_normalize(pooled.reshape(1, -1))
