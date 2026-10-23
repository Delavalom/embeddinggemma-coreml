import torch
from upstream import RMSNorm as UpstreamRMSNorm

from egcm.model.norm import RMSNormCF


def to_cf(x):  # (S, C) -> (1, C, 1, S)
    return x.T.reshape(1, x.shape[1], 1, x.shape[0])


def from_cf(y):  # (1, C, 1, S) -> (S, C)
    return y.reshape(y.shape[1], y.shape[3]).T


def test_matches_upstream_with_scale():
    torch.manual_seed(0)
    x = torch.randn(7, 512) * 30
    w = torch.rand(512) + 0.5
    ref = UpstreamRMSNorm(512)
    ref.weight.data = w.clone()
    n = RMSNormCF(512, with_scale=True)
    n.weight.data = w.clone()
    assert torch.allclose(from_cf(n(to_cf(x))), ref(x), rtol=1e-5, atol=1e-6)


def test_matches_upstream_without_scale():
    torch.manual_seed(1)
    x = torch.randn(5, 256) * 3
    n = RMSNormCF(256, with_scale=False)
    assert not hasattr(n, "weight")
    assert torch.allclose(from_cf(n(to_cf(x))), UpstreamRMSNorm(256, with_scale=False)(x), rtol=1e-5, atol=1e-6)


def test_scale_safe_identical_in_fp32():
    torch.manual_seed(2)
    x = torch.randn(1, 512, 1, 9) * 4000
    a, b = RMSNormCF(512, with_scale=False), RMSNormCF(512, with_scale=False, prescale=256.0)
    assert torch.allclose(a(x), b(x), rtol=1e-5, atol=1e-6)


def test_scale_safe_fp16_matches_fp32():
    torch.manual_seed(3)
    x = torch.randn(1, 512, 1, 9) * 4000  # x² overflows fp16 (max 65504) without the prescale
    want = RMSNormCF(512, with_scale=False)(x)
    got = RMSNormCF(512, with_scale=False, prescale=256.0).half()(x.half())
    assert torch.isfinite(got).all()
    assert torch.allclose(got.float(), want, atol=1e-2)
    plain = RMSNormCF(512, with_scale=False).half()(x.half())
    assert not torch.allclose(plain.float(), want, atol=1e-2)  # the failure the prescale exists to fix


def test_prescale_keeps_eps_exact():
    """eps must be scaled to eps/s². Tiny inputs (mean(x²) << eps) make eps dominate, so using plain eps would
    change the output by about a factor s. s <= 4 keeps eps/s² above the fp16 floor, so no floor applies."""
    torch.manual_seed(4)
    x = torch.randn(5, 512) * 1e-4
    assert x.pow(2).mean() < 1e-6 / 50
    want = UpstreamRMSNorm(512, eps=1e-6, with_scale=False)(x)
    for s in (2.0, 4.0):
        got = from_cf(RMSNormCF(512, eps=1e-6, with_scale=False, prescale=s)(to_cf(x)))
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-7), s


def test_eps_floor():
    assert RMSNormCF(8, eps=1e-6).eps_eff == 1e-6  # s = 1: unchanged
    assert RMSNormCF(8, eps=1e-6, prescale=4.0).eps_eff == 1e-6 / 16
    assert RMSNormCF(8, eps=1e-6, prescale=256.0).eps_eff == 2.0**-24  # 1e-6 / 65536 would be 0 in fp16


def test_scale_safe_fp16_zero_input():
    y = RMSNormCF(512, with_scale=False, prescale=256.0).half()(torch.zeros(1, 512, 1, 3, dtype=torch.half))
    assert torch.isfinite(y).all()
    assert (y == 0).all()


def test_scale_safe_fp16_small_input():
    """fp16 agrees with the same prescaled module in fp32. Both differ from the exact norm here (rms 0.01 is far
    below what s = 256 suits, see RMSNormCF); this checks finiteness and fp16 agreement only."""
    torch.manual_seed(5)
    x = torch.randn(1, 512, 1, 9) * 0.01
    n = RMSNormCF(512, with_scale=False, prescale=256.0)
    want = n(x)
    got = n.half()(x.half())
    assert torch.isfinite(got).all()
    assert torch.allclose(got.float(), want, rtol=0.05, atol=0.01 * want.abs().max().item())
