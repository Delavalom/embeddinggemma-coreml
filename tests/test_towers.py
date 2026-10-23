"""The rewrite's towers and embedders against the upstream towers (tests/upstream_towers.py), on a tiny model
whose weights go through the real mapping in egcm.weights."""

import pytest
import tiny
import torch
import upstream as up
import upstream_towers as ut

from egcm.model.masks import key_mask
from egcm.model.text import rope_tables

torch.set_grad_enabled(False)


@pytest.fixture(scope="module")
def models():
    text, vision, mm = tiny.reference()
    sd = tiny.hf_state_dict(text, vision, mm)
    return (text, vision, mm), sd, tiny.rewrite(sd)


def l2(v):
    return v / v.norm(dim=-1, keepdim=True)


def close(got, want, rel=1e-5):
    err = (got - want).abs().max().item()
    assert err <= rel * want.abs().max().item(), (err, want.abs().max().item())


def test_text_embedding_matches_upstream_with_padding(models):
    (text, _, _), _, (text_emb, _, _) = models
    ids = torch.tensor([[2, 11, 17, 3, 29, 8, 1]])
    want = l2(text(text.embed(ids)).mean(1))  # unpadded upstream, mean over all tokens

    n, S = ids.shape[1], text_emb.seq_len
    padded = torch.zeros(1, S, dtype=torch.int32)
    padded[0, :n] = ids[0]
    mask = (torch.arange(S) < n).float()[None]
    got = text_emb(padded, mask)
    assert got.shape == (1, tiny.TEXT.out_dim)
    close(got, want)


def test_text_tokens_and_ple_match_upstream(models):
    (text, _, _), _, (text_emb, _, table) = models
    ids = torch.tensor([[2, 9, 33, 4, 1]])
    emb_up = text.embed(ids)
    emb = table[ids[0]].T[None, :, None, :]
    assert torch.equal(emb[0, :, 0].T, emb_up[0])  # same scaled lookup, bit for bit

    ple_up = text.ple(emb_up)[0]  # (S, L, P)
    ple = torch.stack([p[0, :, 0].T for p in text_emb.tower.per_layer_inputs(emb)], 1)
    close(ple, ple_up)

    S = ids.shape[1]
    got = text_emb.tower(emb, None, rope_tables(tiny.TEXT, S))
    close(got[0, :, 0].T, text(emb_up)[0])


def test_image_embedding_matches_upstream(models):
    (text, vision, mm), _, (_, image_emb, _) = models
    torch.manual_seed(1)
    pixels = torch.rand(1, 3, tiny.BUCKET.height, tiny.BUCKET.width)
    hp, wp = image_emb.vision.hp, image_emb.vision.wp
    patches = ut.convert_image_to_patches(pixels[0], tiny.VISION.patch)[None]
    pos = up.patch_positions(hp, wp)[None]
    soft = mm(vision(patches, pos))  # (1, tokens, C_text), unscaled
    seq = torch.cat([text.embed(torch.tensor([tiny.PREFIX])), soft, text.embed(torch.tensor([tiny.SUFFIX]))], 1)
    want = l2(text(seq).mean(1))

    got = image_emb(pixels)
    assert image_emb.seq_len == seq.shape[1]
    close(got, want)
    close(image_emb.vision(pixels)[0, :, 0].T, soft[0])


def test_patch_embedding_folds_the_rescale(models):
    (_, vision, _), _, (_, image_emb, _) = models
    torch.manual_seed(2)
    pixels = torch.rand(1, 3, tiny.BUCKET.height, tiny.BUCKET.width)
    v = image_emb.vision
    patches = ut.convert_image_to_patches(pixels[0], tiny.VISION.patch)[None]
    pos = up.patch_positions(v.hp, v.wp)[None]
    want = vision.patch_embedder(patches, pos, torch.zeros(1, v.patches, dtype=torch.bool))[0]
    close(v.embed(pixels)[0, :, 0].T, want)


def test_padded_keys_do_not_change_valid_tokens(models):
    _, _, (text_emb, _, table) = models
    ids = torch.tensor([2, 5, 6, 1, 0, 0, 0, 0, 0, 0])
    emb = table[ids].T[None, :, None, :]
    rope, mask = rope_tables(tiny.TEXT, 10), key_mask((torch.arange(10) < 4).float()[None])
    full = text_emb.tower(emb, mask, rope)
    emb2 = emb.clone()
    emb2[..., 4:] = torch.randn_like(emb2[..., 4:]) * 100  # garbage in the padding
    noisy = text_emb.tower(emb2, mask, rope)
    close(noisy[..., :4], full[..., :4], rel=1e-6)
