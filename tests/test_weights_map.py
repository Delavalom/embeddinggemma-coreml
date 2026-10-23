"""egcm.weights: the HF key mapping (against s0's keys.json and the tiny model), strict loading, the logit bound."""

import json

import pytest
import tiny
import torch

from egcm.model.attention import AttentionCF
from egcm.paths import ARTIFACTS
from egcm.weights import LOGIT_BOUND, build_text_tower, build_vision_encoder, check_logit_bound, mapping_for_keys

KEYS = ARTIFACTS / "s0_reference" / "keys.json"


@pytest.mark.skipif(not KEYS.exists(), reason="run `just stage 0` first")
def test_every_needed_key_mapped_and_no_audio():
    keys = json.loads(KEYS.read_text())
    used, unused = mapping_for_keys(keys)
    assert unused and all(k.split(".")[0] in ("audio_tower", "embed_audio") for k in unused), unused[:10]
    assert any(k.startswith("language_model.layers.23.") for k in used)
    assert any("vision_tower.encoder.layers.15." in k for k in used)
    assert not [k for k in used if "audio" in k]


@pytest.fixture(scope="module")
def sd():
    return tiny.hf_state_dict(*tiny.reference())


def test_tiny_model_uses_every_key(sd):
    used, unused = mapping_for_keys(sd)
    assert not unused
    assert len(used) == len(sd)


def test_model_prefix_is_stripped():
    assert mapping_for_keys(["model.language_model.norm.weight"]) == (["model.language_model.norm.weight"], [])


@pytest.mark.parametrize(
    "drop", ["language_model.layers.1.layer_scalar", "language_model.ple.per_layer_model_projection.weight"])
def test_missing_text_key_fails_strict_load(sd, drop):
    partial = {k: v for k, v in sd.items() if k != drop}
    with pytest.raises(RuntimeError, match="Missing key"):
        build_text_tower(tiny.TEXT, partial)


def test_missing_vision_key_fails_strict_load(sd):
    partial = {k: v for k, v in sd.items() if "encoder.layers.1.mlp.up_proj" not in k}
    with pytest.raises(RuntimeError, match="Missing key"):
        build_vision_encoder(tiny.VISION, partial, tiny.TEXT.hidden)


def test_extra_layer_fails_strict_load(sd):
    extra = dict(sd)
    extra["language_model.layers.3.input_layernorm.weight"] = torch.ones(tiny.TEXT.hidden)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        build_text_tower(tiny.TEXT, extra)


def _attn(q_max: float, k_max: float, head_dim: int = 16) -> AttentionCF:
    m = AttentionCF(32, 2, 1, head_dim, 1e-6, rope_halves=1)
    m.q_norm.weight.data = torch.linspace(-q_max, q_max / 2, head_dim)  # max |w| = q_max, carried by a negative
    m.k_norm.weight.data = torch.full((head_dim,), k_max)
    return m


def test_logit_bound_at_the_limit_passes():
    got = check_logit_bound(torch.nn.ModuleList([_attn(2.0, 2.0)]))  # 16 * 2 * 2 = 64
    assert got == {"0": pytest.approx(LOGIT_BOUND)}


def test_logit_bound_above_the_limit_raises():
    with pytest.raises(ValueError, match="logit bound"):
        check_logit_bound(torch.nn.ModuleList([_attn(1.0, 1.0), _attn(2.0, 2.01)]))


def test_logit_bound_checked_when_loading(sd):
    big = dict(sd)
    big["language_model.layers.1.self_attn.q_norm.weight"] = torch.full((16,), 10.0)  # 16 * 10 * 1.5 > 64
    with pytest.raises(ValueError, match="logit bound"):
        build_text_tower(tiny.TEXT, big)
