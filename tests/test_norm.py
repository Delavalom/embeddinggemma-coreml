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
