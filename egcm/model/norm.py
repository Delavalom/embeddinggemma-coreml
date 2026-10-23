import torch
from torch import nn

# Smallest fp16 subnormal. eps / s² is floored here so it never rounds to 0 in fp16 (which would turn an all-zero
# token into 0 * rsqrt(0) = NaN).
FP16_EPS_FLOOR = 2.0**-24


class RMSNormCF(nn.Module):
    """RMSNorm over dim 1 of a channels-first (B, C, 1, S) tensor.

    Matches upstream `Gemma4RMSNorm` / `EmbeddingGemma2RMSNorm`: `x * (mean(x²) + eps)^-0.5 * weight`, with the
    weight used as-is (no `1 + w`). `prescale` (s) computes `x/s * rsqrt(mean((x/s)²) + eps_eff)`, which keeps x²
    inside the fp16 range. With `eps_eff = eps/s²` this is the same value as the plain form in exact maths.

    `eps_eff = max(eps/s², 2**-24)`. The floor is a no-op for s <= 4 (eps = 1e-6). Otherwise it is exact apart
    from a relative error of about 2**-24 / (2 * mean((x/s)²)): negligible while mean((x/s)²) is well above 1e-7,
    and dominant below it. In raw units that bound is mean(x²) ~ s² * 1e-7, so for s = 256 a token with rms 1 is
    off by ~0.2%, rms 0.1 by ~16%, rms 0.01 by ~85%. A prescale therefore only suits norms whose every token is
    large; choose s per norm from the probe (s3) against the smallest token rms that norm sees, not just the largest.
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True, prescale: float = 1.0):
        super().__init__()
        self.eps, self.prescale, self.with_scale = eps, prescale, with_scale
        self.eps_eff = max(eps / (prescale * prescale), FP16_EPS_FLOOR)
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.prescale
        xs = x / s if s != 1.0 else x
        y = xs * torch.rsqrt(xs.pow(2).mean(1, keepdim=True) + self.eps_eff)
        if self.with_scale:
            y = y * self.weight.view(1, -1, 1, 1)
        return y
