import torch
from torch import nn


class RMSNormCF(nn.Module):
    """RMSNorm over dim 1 of a channels-first (B, C, 1, S) tensor.

    Matches upstream `Gemma4RMSNorm` / `EmbeddingGemma2RMSNorm`: `x * (mean(x²) + eps)^-0.5 * weight`, with the
    weight used as-is (no `1 + w`). `prescale` (s) computes `x/s * rsqrt(mean((x/s)²) + eps/s²)`, which is the same
    value in exact maths but keeps x² inside the fp16 range. s = 1 is the plain form.
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True, prescale: float = 1.0):
        super().__init__()
        self.eps, self.prescale, self.with_scale = eps, prescale, with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.prescale
        xs = x / s if s != 1.0 else x
        y = xs * torch.rsqrt(xs.pow(2).mean(1, keepdim=True) + self.eps / (s * s))
        if self.with_scale:
            y = y * self.weight.view(1, -1, 1, 1)
        return y
