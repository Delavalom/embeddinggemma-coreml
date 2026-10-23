"""RoPE tables against the upstream code copied into tests/upstream.py.

Our sin tables carry the rotate_half sign (see egcm/model/rope.py), so they are compared with upstream's sin after
negating the first block of each rotated pair; the application tests compare final results, which is what matters.
"""

import pytest
import torch
import upstream as up

from egcm.model.rope import apply_rope_cf, rope_1d, rope_2d


def signed(sin, blocks):  # upstream (S, D) sin -> expected signed table: blocks with even index negated
    parts = sin.chunk(blocks, dim=-1)
    return torch.cat([-p if i % 2 == 0 else p for i, p in enumerate(parts)], -1)


@pytest.mark.parametrize(("S", "D", "theta"), [(11, 256, 1e4), (290, 512, 1e6)])
def test_1d_matches_upstream(S, D, theta):
    torch.manual_seed(0)
    cos_ref, sin_ref = up.text_cos_sin(up.text_inv_freq(theta, D), torch.arange(S)[None])  # (1, S, D)
    q, k = torch.randn(1, 4, S, D), torch.randn(1, 2, S, D)  # upstream (B, H, S, D)
    q_ref, k_ref = up.apply_rotary_pos_emb_qk(q, k, cos_ref, sin_ref)

    cos, sin = rope_1d(S, D, theta)
    assert cos.shape == sin.shape == (1, D, 1, S)
    assert torch.allclose(cos.reshape(D, S).T, cos_ref[0], atol=1e-6)
    assert torch.allclose(sin.reshape(D, S).T, signed(sin_ref[0], 2), atol=1e-6)
    for x, ref in ((q, q_ref), (k, k_ref)):
        x_cf = x[0].permute(0, 2, 1).unsqueeze(2)  # (H, D, 1, S): one head per batch row
        out = apply_rope_cf(x_cf, cos, sin, halves=1)
        assert torch.allclose(out.squeeze(2).permute(0, 2, 1), ref[0], atol=1e-5)


@pytest.mark.parametrize(("Hp", "Wp"), [(6, 9), (33, 15)])
def test_2d_matches_upstream(Hp, Wp):
    torch.manual_seed(0)
    D, theta, heads = 64, 100.0, 3
    pos = up.patch_positions(Hp, Wp)[None]  # (1, S, 2)
    cos_ref, sin_ref = up.vision_cos_sin(up.axial_inv_freq(theta, D), pos)  # (1, S, D)
    x = torch.randn(1, Hp * Wp, heads, D)  # upstream (B, S, H, D) before the transpose
    ref = up.apply_multidimensional_rope(x, cos_ref, sin_ref, pos)

    cos, sin = rope_2d(Hp, Wp, D, theta)
    assert cos.shape == sin.shape == (1, D, 1, Hp * Wp)
    assert torch.allclose(cos.reshape(D, -1).T, cos_ref[0], atol=1e-6)
    assert torch.allclose(sin.reshape(D, -1).T, signed(sin_ref[0], 4), atol=1e-6)
    x_cf = x[0].permute(1, 2, 0).unsqueeze(2)  # (H, D, 1, S)
    out = apply_rope_cf(x_cf, cos, sin, halves=2)
    assert torch.allclose(out.squeeze(2).permute(2, 0, 1), ref[0], atol=1e-5)


def test_2d_is_not_1d():
    """Guard against the test passing trivially: the axial split must change the result."""
    torch.manual_seed(0)
    cos, sin = rope_2d(4, 5, 64, 100.0)
    x = torch.randn(1, 64, 1, 20)
    assert not torch.allclose(apply_rope_cf(x, cos, sin, halves=2), apply_rope_cf(x, cos, sin, halves=1))
