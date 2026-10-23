"""RoPE cos/sin tables in the channels-first layout (1, D, 1, S), and their application.

Text (`EmbeddingGemma2RotaryEmbedding`): positions 0..S-1, `inv_freq = 1/theta^(arange(0, D, 2)/D)`,
table `cat(f, f)`, `rotate_half` over the whole head.

Vision (`Gemma4VisionRotaryEmbedding` + `apply_multidimensional_rope`): patch k of an Hp x Wp grid is at
(x, y) = (k % Wp, k // Wp) (the processor's row-major meshgrid). `inv_freq = 1/theta^(arange(0, D/2, 2)/(D/2))`,
table `cat(fx, fx, fy, fy)`. The head is split into an x half and a y half and `rotate_half` runs in each.
The tables are built once per shape and baked into the converted model as constants.
"""

import torch


def _cf(table: torch.Tensor) -> torch.Tensor:  # (S, D) -> (1, D, 1, S)
    S, D = table.shape
    return table.T.reshape(1, D, 1, S).contiguous()


def rope_1d(S: int, D: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float) / D))
    f = torch.arange(S, dtype=torch.float)[:, None] * inv
    e = torch.cat([f, f], -1)  # (S, D)
    return _cf(e.cos()), _cf(e.sin())


def rope_2d(Hp: int, Wp: int, D: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.arange(Wp, dtype=torch.float).repeat(Hp)
    ys = torch.arange(Hp, dtype=torch.float).repeat_interleave(Wp)
    half = D // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, 2, dtype=torch.float) / half))
    fx, fy = xs[:, None] * inv, ys[:, None] * inv
    cos = torch.cat([fx.cos(), fx.cos(), fy.cos(), fy.cos()], -1)  # (S, D)
    sin = torch.cat([fx.sin(), fx.sin(), fy.sin(), fy.sin()], -1)
    return _cf(cos), _cf(sin)


def _rotate_half_cf(x: torch.Tensor) -> torch.Tensor:  # rotate_half along dim 1
    h = x.shape[1] // 2
    return torch.cat((-x[:, h:], x[:, :h]), 1)


def apply_rope_cf(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, halves: int) -> torch.Tensor:
    """x: (B, D, 1, S) for one head. halves=1: standard RoPE; halves=2: axial (x half, then y half)."""
    if halves == 1:
        return x * cos + _rotate_half_cf(x) * sin
    n = x.shape[1] // halves
    parts = []
    for i in range(halves):
        sl = slice(i * n, (i + 1) * n)
        parts.append(x[:, sl] * cos[:, sl] + _rotate_half_cf(x[:, sl]) * sin[:, sl])
    return torch.cat(parts, 1)
