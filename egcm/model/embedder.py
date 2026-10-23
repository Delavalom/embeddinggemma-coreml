"""The Core ML entry points: token ids or pixels in, one L2-normalized embedding (1, out_dim) out.

Pooling is sentence-transformers' mean over the attention mask (`include_prompt: true`, so the prompt, bos and eos
count), then L2 normalization. Each embedder is built for one fixed sequence length and owns that length's RoPE
tables; the towers are shared, so building many embedders does not copy weights.
"""

import torch
import torch.nn.functional as F
from torch import nn

from egcm.model.masks import key_mask
from egcm.model.text import TextTower, rope_tables
from egcm.model.vision import VisionTower

# Sequence lengths of the text functions (embed_text_<L>); a text uses the smallest that fits.
TEXT_LENGTHS = (16, 32, 64, 128)


def l2_normalize(v: torch.Tensor) -> torch.Tensor:
    return v * torch.rsqrt((v * v).sum(-1, keepdim=True))


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
        pooled = (tokens * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True)
        return l2_normalize(pooled.reshape(1, -1))


class ImageEmbedder(_RopeOwner):
    """pixels (1, 3, H, W) in [0, 1] for one bucket -> (1, out_dim).

    The sequence is `prefix + soft tokens + suffix`: the prefix and suffix are the fixed non-image tokens of s0's
    recorded layout (bos, <|image> / <image|>, eos), embedded with the scaled table and baked in as constants, so the
    image functions do not carry the embedding table. Every token is valid, so no mask, and pooling is a plain mean.
    """

    def __init__(self, vision: VisionTower, tower: TextTower, prefix_emb: torch.Tensor, suffix_emb: torch.Tensor):
        super().__init__()
        self.vision, self.tower = vision, tower
        self.register_buffer("prefix", prefix_emb, persistent=False)  # (1, C, 1, P)
        self.register_buffer("suffix", suffix_emb, persistent=False)  # (1, C, 1, Q)
        self.seq_len = prefix_emb.shape[-1] + vision.tokens + suffix_emb.shape[-1]  # fixed at construction
        self._register_rope(tower, self.seq_len)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([self.prefix, self.vision(pixels), self.suffix], dim=-1)  # (1, C, 1, seq_len)
        tokens = self.tower(emb, None, self._rope())
        return l2_normalize(tokens.mean(-1, keepdim=True).reshape(1, -1))
