"""Channels-first RMSNorm that stays exact in FP16 on the Neural Engine, at any activation scale."""

import torch
from torch import nn

# Lower bound for the per-token max |x|. Keeps 1/m <= 64, so an all-zero token stays finite (it maps to 0).
MIN_TOKEN_MAX = 2.0**-6


class RMSNormCF(nn.Module):
    """RMSNorm over dim 1 of a channels-first (B, C, 1, S) tensor, scaled per token.

    Same maths as upstream `Gemma4RMSNorm` / `EmbeddingGemma2RMSNorm`, `x * (mean(x²) + eps)^-0.5 * weight`, with
    the weight used as-is (no `1 + w`). It is computed as

        m = max(max_c |x|, 2**-6),  r = 1/m,  xs = x * r,  e = r * sqrt(eps)
        y = xs * rsqrt(mean(xs²) + e²) * weight

    which is identical in exact maths: xs * rsqrt(r² (mean(x²) + eps)) = x * rsqrt(mean(x²) + eps).

    Why this form (FP16 on the Neural Engine):
    - |xs| <= 1, so xs² never overflows (plain x² overflows FP16 once |x| > 256).
    - The ANE flushes FP16 subnormals (below 2**-14 ≈ 6.1e-5) to zero. eps = 1e-6 is subnormal, so `+ eps` would
      be `+ 0` and an all-zero token would give 0 * rsqrt(0) = NaN. sqrt(eps) = 1e-3 is a normal value, and so is
      e = r * sqrt(eps) whenever eps matters (small m, large r). When m is large enough that e² flushes, eps is
      negligible anyway: mean(xs²) >= 1/C because the largest element of xs is 1.
    - The clamp keeps r <= 64, so a zero token gives xs = 0, e² = 4e-3, y = 0.
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps, self.with_scale = eps, with_scale
        self.sqrt_eps = eps**0.5
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.abs().amax(1, keepdim=True).clamp_min(MIN_TOKEN_MAX)  # maximum op; clamp(min=) clips at fp32 max -> inf
        r = 1.0 / m
        xs = x * r
        e = r * self.sqrt_eps
        y = xs * torch.rsqrt(xs.pow(2).mean(1, keepdim=True) + e * e)
        if self.with_scale:
            y = y * self.weight.view(1, -1, 1, 1)
        return y
