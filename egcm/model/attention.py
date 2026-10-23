"""Neural-Engine-layout attention for both towers (Apple ml-ane-transformers style).

Hidden states are (B, C, 1, S); projections are 1x1 convs; heads are split along dim 1 and attended one at a
time with einsums, so no 5-D tensors appear. Same maths as upstream EmbeddingGemma2Attention and
Gemma4VisionAttention: q_norm / k_norm (weighted) and v_norm (unweighted) per head, then RoPE on q and k,
bidirectional attention with scaling = 1.0, GQA by sharing kv head i // (heads // kv_heads).
"""

import torch
from torch import nn

from egcm.model.norm import RMSNormCF
from egcm.model.rope import apply_rope_cf


def conv1x1(cin: int, cout: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 1, bias=False)


class AttentionCF(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, eps: float, rope_halves: int):
        super().__init__()
        if heads % kv_heads:
            raise ValueError(f"heads={heads} is not a multiple of kv_heads={kv_heads}")
        self.heads, self.kv, self.hd, self.halves = heads, kv_heads, head_dim, rope_halves
        self.q, self.k = conv1x1(hidden, heads * head_dim), conv1x1(hidden, kv_heads * head_dim)
        self.v, self.o = conv1x1(hidden, kv_heads * head_dim), conv1x1(heads * head_dim, hidden)
        self.q_norm = RMSNormCF(head_dim, eps)
        self.k_norm = RMSNormCF(head_dim, eps)
        self.v_norm = RMSNormCF(head_dim, eps, with_scale=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor | None):
        """x: (B, C, 1, S). cos, sin: (1, head_dim, 1, S). mask: additive (B, S_k, 1, 1) over keys, or None."""
        qs = [apply_rope_cf(self.q_norm(h), cos, sin, self.halves) for h in self.q(x).split(self.hd, dim=1)]
        # keys go key-major, (B, S_k, 1, hd), so q·k is one einsum producing (B, S_k, 1, S_q)
        ks = [apply_rope_cf(self.k_norm(h), cos, sin, self.halves).transpose(1, 3)
              for h in self.k(x).split(self.hd, dim=1)]
        vs = [self.v_norm(h) for h in self.v(x).split(self.hd, dim=1)]
        rep = self.heads // self.kv
        outs = []
        for i, q in enumerate(qs):
            k, v = ks[i // rep], vs[i // rep]
            w = torch.einsum("bchq,bkhc->bkhq", q, k)  # (B, S_k, 1, S_q); scaling = 1.0
            if mask is not None:
                w = w + mask
            w = w.softmax(dim=1)
            outs.append(torch.einsum("bkhq,bchk->bchq", w, v))  # (B, hd, 1, S_q)
        return self.o(torch.cat(outs, 1))
