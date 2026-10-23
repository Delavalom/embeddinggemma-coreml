import json

import pytest

from egcm import config
from egcm.buckets import MAX_SEQ
from egcm.paths import HF_SNAPSHOT

FULL = (5, 11, 17, 23)


def fake_config(**text_overrides):
    """A config.json with the shape of google/embeddinggemma-2 (only the keys config.load reads)."""
    layer_types = ["full_attention" if i in FULL else "sliding_attention" for i in range(24)]
    text = {
        "hidden_size": 512, "num_hidden_layers": 24, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 256, "intermediate_size": 2048, "hidden_size_per_layer_input": 512, "embedding_dim": 768,
        "vocab_size": 262144, "rms_norm_eps": 1e-6, "pad_token_id": 0, "sliding_window": 512, "attention_bias": False,
        "layer_types": layer_types,
        "per_layer_config": {f"{i:02d}": {"head_dim": 512, "num_key_value_heads": 1} for i in FULL},
        "rope_parameters": {
            "full_attention": {"rope_theta": 1e6, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
        },
    }
    text.update(text_overrides)
    vision = {
        "hidden_size": 768, "num_hidden_layers": 16, "num_attention_heads": 12, "num_key_value_heads": 12,
        "head_dim": 64, "intermediate_size": 3072, "patch_size": 16, "pooling_kernel_size": 3,
        "rope_parameters": {"rope_theta": 100.0, "rope_type": "axial"}, "rms_norm_eps": 1e-6,
        "position_embedding_size": 10240, "standardize": False, "use_clipped_linears": False, "attention_bias": False,
    }
    return {"text_config": text, "vision_config": vision}


def write(tmp_path, c):
    (tmp_path / "config.json").write_text(json.dumps(c))
    return tmp_path


def check_known_values(text, vis):
    assert (text.hidden, text.layers, text.heads, text.ffn) == (512, 24, 4, 2048)
    assert (text.ple_dim, text.out_dim, text.vocab, text.pad_id, text.eps) == (512, 768, 262144, 0, 1e-6)
    assert text.sliding_window == 512
    assert [i for i, t in enumerate(text.layer_types) if t == "full_attention"] == list(FULL)
    assert text.head_dim == {"sliding_attention": 256, "full_attention": 512}
    assert text.kv_heads == {"sliding_attention": 2, "full_attention": 1}
    assert text.theta == {"sliding_attention": 1e4, "full_attention": 1e6}
    assert (vis.hidden, vis.layers, vis.heads, vis.kv_heads, vis.head_dim, vis.ffn) == (768, 16, 12, 12, 64, 3072)
    assert (vis.patch, vis.pool, vis.theta, vis.eps, vis.pos_size) == (16, 3, 100.0, 1e-6, 10240)


def test_load_synthetic(tmp_path):
    text, vis, raw = config.load(write(tmp_path, fake_config()))
    check_known_values(text, vis)
    assert raw["text_config"]["hidden_size"] == 512


def test_inconsistent_full_layers_rejected(tmp_path):
    c = fake_config()
    c["text_config"]["per_layer_config"]["17"]["head_dim"] = 256
    with pytest.raises(ValueError, match="17"):
        config.load(write(tmp_path, c))


def test_override_on_sliding_layer_rejected(tmp_path):
    c = fake_config()
    c["text_config"]["per_layer_config"]["03"] = {"head_dim": 512, "num_key_value_heads": 1}
    with pytest.raises(ValueError, match="03"):
        config.load(write(tmp_path, c))


def test_unsupported_vision_options_rejected(tmp_path):
    c = fake_config()
    c["vision_config"]["standardize"] = True
    with pytest.raises(ValueError, match="standardize"):
        config.load(write(tmp_path, c))


@pytest.mark.parametrize("tower", ["text_config", "vision_config"])
def test_attention_bias_rejected(tmp_path, tower):
    c = fake_config()
    c[tower]["attention_bias"] = True
    with pytest.raises(ValueError, match="attention_bias"):
        config.load(write(tmp_path, c))


def test_sliding_window_shorter_than_max_seq_rejected(tmp_path):
    with pytest.raises(ValueError, match="sliding_window"):
        config.load(write(tmp_path, fake_config(sliding_window=MAX_SEQ - 1)))
    config.load(write(tmp_path, fake_config(sliding_window=MAX_SEQ)))  # equal is fine: every token is in the window


@pytest.mark.skipif(not (HF_SNAPSHOT / "config.json").exists(), reason="HF snapshot not downloaded (run s0)")
def test_load_real_config():
    text, vis, _ = config.load(HF_SNAPSHOT)
    check_known_values(text, vis)


def test_configs_are_hashable_and_read_only(tmp_path):
    text, vis, _ = config.load(write(tmp_path, fake_config()))
    text2, vis2, _ = config.load(write(tmp_path, fake_config()))
    assert hash(text) == hash(text2) and text == text2
    assert hash(vis) == hash(vis2) and vis == vis2
    assert {text, text2} == {text}
    with pytest.raises(TypeError):
        text.head_dim["full_attention"] = 1


@pytest.mark.parametrize("key", ["24", "5", "foo"])
def test_unknown_per_layer_config_key_rejected(tmp_path, key):
    c = fake_config()
    c["text_config"]["per_layer_config"][key] = {"head_dim": 512}
    with pytest.raises(ValueError, match="not layer indices"):
        config.load(write(tmp_path, c))


@pytest.mark.parametrize("layer_types", [["sliding_attention"] * 23, ["sliding_attention"] * 23 + ["chunked"]])
def test_bad_layer_types_rejected(tmp_path, layer_types):
    with pytest.raises(ValueError, match="layer_types"):
        config.load(write(tmp_path, fake_config(layer_types=layer_types)))


def test_text_rope_type_rejected(tmp_path):
    c = fake_config()
    c["text_config"]["rope_parameters"]["full_attention"]["rope_type"] = "yarn"
    with pytest.raises(ValueError, match="rope_type"):
        config.load(write(tmp_path, c))


def test_vision_rope_type_rejected(tmp_path):
    c = fake_config()
    c["vision_config"]["rope_parameters"]["rope_type"] = "default"
    with pytest.raises(ValueError, match="rope_type"):
        config.load(write(tmp_path, c))


def test_clipped_linears_rejected(tmp_path):
    c = fake_config()
    c["vision_config"]["use_clipped_linears"] = True
    with pytest.raises(ValueError, match="use_clipped_linears"):
        config.load(write(tmp_path, c))
