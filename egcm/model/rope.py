"""RoPE cos/sin tables in the channels-first layout (1, D, 1, S), and their application.

Text (`EmbeddingGemma2RotaryEmbedding`): positions 0..S-1, `inv_freq = 1/theta^(arange(0, D, 2)/D)`,
table `cat(f, f)`, `rotate_half` over the whole head.

Vision (`Gemma4VisionRotaryEmbedding` + `apply_multidimensional_rope`): patch k of an Hp x Wp grid is at
(x, y) = (k % Wp, k // Wp) (the processor's row-major meshgrid). `inv_freq = 1/theta^(arange(0, D/2, 2)/(D/2))`,
table `cat(fx, fx, fy, fy)`. The head is split into an x half and a y half and `rotate_half` runs in each.

The sign of `rotate_half` is folded into the sin table (exact): with a head split into blocks a|b,
`rotate_half(x) * sin = cat(-b, a) * cat(sin_a, sin_b) = cat(b, a) * cat(-sin_a, sin_b)`. So the tables returned
here are `(cos, signed_sin)`, and applying them is one chunk, one cat, two muls and an add, with no `mul(-1)`.
For vision the same holds per half, giving blocks a|b|c|d -> cat(b, a, d, c) with sin signs (-, +, -, +).
The tables are built once per shape and baked into the converted model as constants.
"""

import torch


def _cf(table: torch.Tensor) -> torch.Tensor:  # (S, D) -> (1, D, 1, S)
    S, D = table.shape
    return table.T.reshape(1, D, 1, S).contiguous()


def _fold_sign(sin: torch.Tensor, halves: int) -> torch.Tensor:
    """Negate the first block of each rotated pair: (1, D, 1, S) sin -> signed sin for apply_rope_cf."""
    blocks = sin.chunk(2 * halves, dim=1)
    return torch.cat([-b if i % 2 == 0 else b for i, b in enumerate(blocks)], 1)


def rope_1d(S: int, D: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, signed_sin), each (1, D, 1, S), for positions 0..S-1."""
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float) / D))
    f = torch.arange(S, dtype=torch.float)[:, None] * inv
    e = torch.cat([f, f], -1)  # (S, D)
    return _cf(e.cos()), _fold_sign(_cf(e.sin()), 1)


def rope_2d(Hp: int, Wp: int, D: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, signed_sin), each (1, D, 1, Hp*Wp), for the row-major patch grid."""
    xs = torch.arange(Wp, dtype=torch.float).repeat(Hp)
    ys = torch.arange(Hp, dtype=torch.float).repeat_interleave(Wp)
    half = D // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, 2, dtype=torch.float) / half))
    fx, fy = xs[:, None] * inv, ys[:, None] * inv
    cos = torch.cat([fx.cos(), fx.cos(), fy.cos(), fy.cos()], -1)  # (S, D)
    sin = torch.cat([fx.sin(), fx.sin(), fy.sin(), fy.sin()], -1)
    return _cf(cos), _fold_sign(_cf(sin), 2)


def apply_rope_cf(x: torch.Tensor, cos: torch.Tensor, signed_sin: torch.Tensor, halves: int) -> torch.Tensor:
    """x: (B, D, 1, S) for one head; tables from rope_1d (halves=1) or rope_2d (halves=2).

    `halves` is a constructor constant, never derived from a shape, so tracing yields static splits (coremltools
    rejects the `int` ops that `x.shape[1] // 2` would trace to).
    """
    blocks = x.chunk(2 * halves, dim=1)
    swapped = torch.cat([blocks[i ^ 1] for i in range(2 * halves)], 1)  # pairs (a, b) -> (b, a)
    return x * cos + swapped * signed_sin
