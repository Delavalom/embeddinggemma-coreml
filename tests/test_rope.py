"""RoPE tables against upstream code.

The `up_*` functions below are copied from vendor/reference (transformers v5.20.2, Apache-2.0), with the class
plumbing removed (config lookups replaced by arguments, `attention_scaling` = 1.0 as upstream sets it for the
'default' and 'axial' rope types). The maths is unchanged.
"""

import pytest
import torch

from egcm.model.rope import apply_rope_cf, rope_1d, rope_2d


# --- modeling_embedding_gemma2.py -----------------------------------------------------------------------------
def up_text_inv_freq(base, dim):  # EmbeddingGemma2RotaryEmbedding.compute_default_rope_parameters
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    return inv_freq


def up_text_cos_sin(inv_freq, position_ids, attention_scaling=1.0):  # EmbeddingGemma2RotaryEmbedding.forward
    freqs = position_ids[..., None].float() * inv_freq.to(dtype=torch.float)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos, sin


def up_rotate_half(x):  # identical in modeling_embedding_gemma2.py and modeling_gemma4.py
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def up_apply_rotary_pos_emb_qk(q, k, cos, sin, unsqueeze_dim=1):  # modeling_embedding_gemma2.apply_rotary_pos_emb
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (up_rotate_half(q) * sin)
    k_embed = (k * cos) + (up_rotate_half(k) * sin)
    return q_embed, k_embed


# --- image_processing_gemma4.py (Step 5: position ids) ----------------------------------------------------------
def up_patch_positions(patch_height, patch_width):
    patch_grid = torch.meshgrid(torch.arange(patch_width), torch.arange(patch_height), indexing="xy")
    stacked_grid = torch.stack(patch_grid, dim=-1)
    return stacked_grid.reshape(patch_height * patch_width, 2)  # (x, y) per patch, row-major


# --- modeling_gemma4.py ---------------------------------------------------------------------------------------
def up_axial_inv_freq(base, dim):  # Gemma4VisionRotaryEmbedding.compute_axial_rope_parameters
    spatial_dim = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
    return inv_freq


def up_recomposition_frequencies(freq):  # Gemma4VisionRotaryEmbedding.recomposition_frequencies
    # in contrast to other 2D rope modules, interleave grids as H-H-W-W
    freq_h, freq_w = freq[:, :, 0], freq[:, :, 1]
    return torch.cat([freq_h, freq_h, freq_w, freq_w], dim=-1)


def up_vision_cos_sin(inv_freq, position_ids, attention_scaling=1.0):  # Gemma4VisionRotaryEmbedding.forward
    freqs = position_ids[..., None].float() * inv_freq.to(dtype=torch.float)
    cos = freqs.cos() * attention_scaling
    sin = freqs.sin() * attention_scaling
    cos = up_recomposition_frequencies(cos)
    sin = up_recomposition_frequencies(sin)
    return cos, sin


def up_apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):  # modeling_gemma4.apply_rotary_pos_emb
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (up_rotate_half(x) * sin)


def up_apply_multidimensional_rope(x, cos, sin, position_ids, unsqueeze_dim=2):  # modeling_gemma4
    ndim = position_ids.shape[-1]
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
    split_sizes = [num_rotated_channels_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    y_parts = [
        up_apply_rotary_pos_emb(x=x_parts[k], cos=cos_parts[k], sin=sin_parts[k], unsqueeze_dim=unsqueeze_dim)
        for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


# --- tests ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("S", "D", "theta"), [(11, 256, 1e4), (290, 512, 1e6)])
def test_1d_matches_upstream(S, D, theta):
    torch.manual_seed(0)
    cos_ref, sin_ref = up_text_cos_sin(up_text_inv_freq(theta, D), torch.arange(S)[None])  # (1, S, D)
    q, k = torch.randn(1, 4, S, D), torch.randn(1, 2, S, D)  # upstream (B, H, S, D)
    q_ref, k_ref = up_apply_rotary_pos_emb_qk(q, k, cos_ref, sin_ref)

    cos, sin = rope_1d(S, D, theta)
    assert cos.shape == sin.shape == (1, D, 1, S)
    assert torch.allclose(cos.reshape(D, S).T, cos_ref[0], atol=1e-6)
    assert torch.allclose(sin.reshape(D, S).T, sin_ref[0], atol=1e-6)
    for x, ref in ((q, q_ref), (k, k_ref)):
        x_cf = x[0].permute(0, 2, 1).unsqueeze(2)  # (H, D, 1, S): one head per batch row
        out = apply_rope_cf(x_cf, cos, sin, halves=1)
        assert torch.allclose(out.squeeze(2).permute(0, 2, 1), ref[0], atol=1e-5)


@pytest.mark.parametrize(("Hp", "Wp"), [(6, 9), (33, 15)])
def test_2d_matches_upstream(Hp, Wp):
    torch.manual_seed(0)
    D, theta, heads = 64, 100.0, 3
    pos = up_patch_positions(Hp, Wp)[None]  # (1, S, 2)
    cos_ref, sin_ref = up_vision_cos_sin(up_axial_inv_freq(theta, D), pos)  # (1, S, D)
    x = torch.randn(1, Hp * Wp, heads, D)  # upstream (B, S, H, D) before the transpose
    ref = up_apply_multidimensional_rope(x, cos_ref, sin_ref, pos)

    cos, sin = rope_2d(Hp, Wp, D, theta)
    assert cos.shape == sin.shape == (1, D, 1, Hp * Wp)
    assert torch.allclose(cos.reshape(D, -1).T, cos_ref[0], atol=1e-6)
    assert torch.allclose(sin.reshape(D, -1).T, sin_ref[0], atol=1e-6)
    x_cf = x[0].permute(1, 2, 0).unsqueeze(2)  # (H, D, 1, S)
    out = apply_rope_cf(x_cf, cos, sin, halves=2)
    assert torch.allclose(out.squeeze(2).permute(2, 0, 1), ref[0], atol=1e-5)


def test_2d_is_not_1d():
    """Guard against the test passing trivially: the axial split must change the result."""
    torch.manual_seed(0)
    cos, sin = rope_2d(4, 5, 64, 100.0)
    x = torch.randn(1, 64, 1, 20)
    assert not torch.allclose(apply_rope_cf(x, cos, sin, halves=2), apply_rope_cf(x, cos, sin, halves=1))
