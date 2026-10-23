import pytest
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


# rms of the input tokens, from far below eps (1e-6 vs mean(x²) = 1e-8) to the largest residual streams seen
SCALES = [1e-4, 1e-2, 1.0, 100.0, 4000.0]


def tokens(rms, S=9, C=512, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(S, C) * rms
    x[0, 3] = 6 * rms  # an outlier channel, as residual streams have
    return x


@pytest.mark.parametrize("rms", SCALES)
def test_matches_upstream_fp32_across_scales(rms):
    x = tokens(rms)
    w = torch.rand(512) + 0.5
    ref = UpstreamRMSNorm(512)
    ref.weight.data = w.clone()
    n = RMSNormCF(512)
    n.weight.data = w.clone()
    want, got = ref(x), from_cf(n(to_cf(x)))
    assert (got - want).abs().max() <= 1e-5 * want.abs().max()


# CPU fp16 is the strict check available in a unit test: torch keeps fp16 subnormals on the CPU, the ANE flushes
# them. The dynamic scaling is built so the ANE never needs a subnormal (see RMSNormCF); the on-device verdict is
# s3's cosine gate on the real model, not this test.
@pytest.mark.parametrize("rms", SCALES)
def test_fp16_error_bounded_across_scales(rms):
    x = tokens(rms)
    want = UpstreamRMSNorm(512, with_scale=False)(x)
    got = from_cf(RMSNormCF(512, with_scale=False).half()(to_cf(x).half())).float()
    assert torch.isfinite(got).all()
    assert (got - want).abs().max() <= 5e-3 * want.abs().max()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_zero_token_stays_finite(dtype):
    x = torch.randn(1, 512, 1, 3)
    x[..., 1] = 0  # one all-zero token among normal ones
    y = RMSNormCF(512, with_scale=False).to(dtype)(x.to(dtype))
    assert torch.isfinite(y).all()
    assert (y[..., 1] == 0).all()


@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_eps_semantics(eps):
    """Tiny inputs (mean(x²) = 1e-10, far below eps) make eps dominate, so a wrongly scaled eps changes the output
    by orders of magnitude. Each eps must match the upstream norm with that eps."""
    x = tokens(1e-5)
    want = UpstreamRMSNorm(512, eps=eps, with_scale=False)(x)
    got = from_cf(RMSNormCF(512, eps=eps, with_scale=False)(to_cf(x)))
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-9)
