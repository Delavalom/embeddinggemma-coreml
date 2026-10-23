import torch

from egcm.model.attention import AttentionCF
from egcm.model.masks import MASK_NEG, key_mask
from egcm.model.rope import rope_1d


def test_key_mask_shape_and_values():
    m = key_mask(torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]]))
    assert m.shape == (1, 5, 1, 1)
    assert m.flatten().tolist() == [0.0, 0.0, 0.0, MASK_NEG, MASK_NEG]
    assert torch.isfinite(m.half()).all()  # fp16-safe, unlike finfo(float32).min


def test_padded_keys_are_ignored():
    """Attention over a padded sequence with key_mask equals attention over the valid prefix alone."""
    torch.manual_seed(0)
    a = AttentionCF(hidden=64, heads=2, kv_heads=1, head_dim=32, eps=1e-6, rope_halves=1)
    for p in a.parameters():
        p.data.normal_(0, 0.2)
    S, valid = 8, 5
    x = torch.randn(1, 64, 1, S)
    cos, sin = rope_1d(S, 32, 1e4)
    mask = key_mask((torch.arange(S) < valid).float()[None])
    padded = a(x, cos, sin, mask)[..., :valid]
    alone = a(x[..., :valid], cos[..., :valid], sin[..., :valid], None)
    assert torch.allclose(padded, alone, atol=1e-5)
