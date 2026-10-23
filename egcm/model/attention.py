"""Neural-Engine-layout attention for both towers (Apple ml-ane-transformers style).

Hidden states are (B, C, 1, S); projections are 1x1 convs; heads are split along dim 1 with `chunk` by a
constructor constant and attended one at a time with einsums, so no 5-D tensors and no shape-derived ints appear
in the traced graph.
"""

import torch
from torch import nn

from egcm.model.norm import RMSNormCF
from egcm.model.rope import apply_rope_cf


def conv1x1(cin: int, cout: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 1, bias=False)


def _split_heads(t: torch.Tensor, n: int) -> tuple[torch.Tensor, ...]:
    # coremltools 9.0 cannot convert chunk(1) (its constantchunk returns a single Var), and full layers have 1 kv head
    return t.chunk(n, dim=1) if n > 1 else (t,)


class AttentionCF(nn.Module):
    """Bidirectional multi-head attention, the maths of upstream EmbeddingGemma2Attention / Gemma4VisionAttention.

    Order: q/k/v projections -> per head q_norm / k_norm (weighted) and v_norm (unweighted) -> RoPE on q and k ->
    softmax(q·k + mask) with scaling = 1.0 -> weighted sum of v -> o projection. GQA: query head i uses kv head
    i // (heads // kv_heads), as upstream `repeat_kv` does. `rope_halves` is 1 for text, 2 for the vision axial RoPE.
    """

    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, eps: float, rope_halves: int):
        super().__init__()
        if heads % kv_heads:
            raise ValueError(f"heads={heads} is not a multiple of kv_heads={kv_heads}")
        self.heads, self.kv_heads, self.head_dim, self.rope_halves = heads, kv_heads, head_dim, rope_halves
        self.q, self.k = conv1x1(hidden, heads * head_dim), conv1x1(hidden, kv_heads * head_dim)
        self.v, self.o = conv1x1(hidden, kv_heads * head_dim), conv1x1(heads * head_dim, hidden)
        self.q_norm = RMSNormCF(head_dim, eps)
        self.k_norm = RMSNormCF(head_dim, eps)
        self.v_norm = RMSNormCF(head_dim, eps, with_scale=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor | None):
        """x: (B, C, 1, S). cos, sin: (1, head_dim, 1, S) from rope_1d / rope_2d. mask: (B, S_k, 1, 1) or None."""
        halves = self.rope_halves
        qs = [apply_rope_cf(self.q_norm(h), cos, sin, halves) for h in _split_heads(self.q(x), self.heads)]
        # keys go key-major, (B, S_k, 1, head_dim), so q·k is one einsum producing (B, S_k, 1, S_q)
        ks = [apply_rope_cf(self.k_norm(h), cos, sin, halves).transpose(1, 3)
              for h in _split_heads(self.k(x), self.kv_heads)]
        vs = [self.v_norm(h) for h in _split_heads(self.v(x), self.kv_heads)]
        rep = self.heads // self.kv_heads
        outs = []
        for i, q in enumerate(qs):
            k, v = ks[i // rep], vs[i // rep]
            w = torch.einsum("bchq,bkhc->bkhq", q, k)  # (B, S_k, 1, S_q); scaling = 1.0
            if mask is not None:
                w = w + mask
            w = w.softmax(dim=1)
            outs.append(torch.einsum("bkhq,bchk->bchq", w, v))  # (B, head_dim, 1, S_q)
        return self.o(torch.cat(outs, 1))
