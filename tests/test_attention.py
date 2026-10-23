"""AttentionCF against EmbeddingGemma2Attention / Gemma4VisionAttention (copied into tests/upstream.py).

Upstream order, both towers: project -> view per head -> q_norm / k_norm / v_norm -> RoPE (q, k) -> eager
attention with scaling=1.0 and GQA repeat_kv -> o_proj.
"""

import pytest
import torch
import upstream as up

from egcm.model.attention import AttentionCF
from egcm.model.rope import rope_1d, rope_2d

NEG = torch.finfo(torch.float32).min  # what upstream's bidirectional mask puts on padded keys in eager mode


def make_pair(ref_cls, hidden, heads, kv_heads, head_dim, halves):
    torch.manual_seed(0)
    ref = ref_cls(hidden, heads, kv_heads, head_dim, eps=1e-6)
    for name, p in ref.named_parameters():
        if name.endswith("norm.weight"):
            p.data = torch.rand_like(p) + 0.5
        else:
            p.data.normal_(0, 0.05)
    m = AttentionCF(hidden, heads, kv_heads, head_dim, eps=1e-6, rope_halves=halves)
    for proj in ("q", "k", "v", "o"):
        getattr(m, proj).weight.data = getattr(ref, f"{proj}_proj").weight.data[:, :, None, None].clone()
    m.q_norm.weight.data = ref.q_norm.weight.data.clone()
    m.k_norm.weight.data = ref.k_norm.weight.data.clone()
    return ref, m


def assert_close(got, want, rel=1e-5):
    err = (got - want).abs().max().item()
    assert err <= rel * want.abs().max().item(), err


def to_cf(x):  # (1, S, C) -> (1, C, 1, S)
    return x[0].T.reshape(1, x.shape[2], 1, x.shape[1])


@pytest.mark.parametrize(("heads", "kv_heads", "head_dim", "theta"), [(4, 2, 256, 1e4), (4, 1, 512, 1e6)])
def test_text_matches_upstream_with_padding(heads, kv_heads, head_dim, theta):
    ref, m = make_pair(up.TextAttention, 512, heads, kv_heads, head_dim, halves=1)
    S, valid = 9, 7
    x = torch.randn(1, S, 512)
    keep = torch.arange(S) < valid
    up_mask = torch.where(keep, 0.0, NEG).view(1, 1, 1, S)  # (B, 1, S_q, S_k), broadcast over queries
    pe = up.text_cos_sin(up.text_inv_freq(theta, head_dim), torch.arange(S)[None])
    want, _ = ref(x, attention_mask=up_mask, position_embeddings=pe)

    cos, sin = rope_1d(S, head_dim, theta)
    mask = torch.where(keep, 0.0, -1e4).view(1, S, 1, 1)  # (B, S_k, 1, 1): fp16-safe
    got = m(to_cf(x), cos, sin, mask)
    assert got.shape == (1, 512, 1, S)
    # fp32 summation order differs from upstream's matmul; with scaling=1.0 and head_dim 512 the logits are large,
    # so allow ~1e-5 relative to the output scale (a real error, e.g. a swapped RoPE half, is O(1)).
    assert_close(got.reshape(512, S).T, want[0])


def test_text_padding_is_ignored():
    _, m = make_pair(up.TextAttention, 512, 4, 2, 256, halves=1)
    S, valid = 9, 6
    x = torch.randn(1, 512, 1, S)
    x2 = x.clone()
    x2[..., valid:] = torch.randn(1, 512, 1, S - valid) * 100
    cos, sin = rope_1d(S, 256, 1e4)
    mask = torch.where(torch.arange(S) < valid, 0.0, -1e4).view(1, S, 1, 1)
    assert torch.allclose(m(x, cos, sin, mask)[..., :valid], m(x2, cos, sin, mask)[..., :valid], atol=1e-5)


def test_vision_matches_upstream():
    ref, m = make_pair(up.VisionAttention, 768, 12, 12, 64, halves=2)
    Hp, Wp = 3, 5
    x = torch.randn(1, Hp * Wp, 768)
    pos = up.patch_positions(Hp, Wp)[None]
    pe = up.vision_cos_sin(up.axial_inv_freq(100.0, 64), pos)
    want, _ = ref(x, position_embeddings=pe, attention_mask=None, position_ids=pos)

    cos, sin = rope_2d(Hp, Wp, 64, 100.0)
    got = m(to_cf(x), cos, sin, None)
    assert_close(got.reshape(768, -1).T, want[0])
