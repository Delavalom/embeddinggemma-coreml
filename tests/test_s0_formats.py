"""s0 output formats and the pixel contract helpers, without the model."""

import json

import numpy as np
import pytest
import torch

from egcm.fixtures import U8_SCALE, load_bucket_pixels, to_unit
from stages.s0_reference import artifacts, pixels
from stages.s0_reference.build_fixtures import manifest_changes


def convert_image_to_patches(image: "torch.Tensor", patch_size: int) -> "torch.Tensor":
    """Copied verbatim from vendor/reference/image_processing_gemma4.py (transformers v5.20.2)."""
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.permute(1, 3, 2, 4, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


def test_patchify_matches_upstream_bit_for_bit():
    u8 = np.random.default_rng(0).integers(0, 256, (96, 144, 3), dtype=np.uint8)
    upstream = convert_image_to_patches(torch.from_numpy(u8).permute(2, 0, 1).float() * U8_SCALE.item(), 16)
    assert np.array_equal(pixels.patchify(u8), upstream.numpy())
    with pytest.raises(ValueError):
        pixels.patchify(u8[:95])


def test_patch_positions_are_the_processor_meshgrid():
    h, w = 48, 80
    gx, gy = torch.meshgrid(torch.arange(w // 16), torch.arange(h // 16), indexing="xy")
    expected = torch.stack([gx, gy], dim=-1).reshape(-1, 2).numpy()
    assert np.array_equal(pixels.patch_positions(h, w), expected)
    assert pixels.patch_positions(h, w)[:4].tolist() == [[0, 0], [1, 0], [2, 0], [3, 0]]


def test_to_unit_is_multiplication_by_f32_reciprocal():
    u8 = np.arange(256, dtype=np.uint8)
    assert np.array_equal(to_unit(u8), u8.astype(np.float32) * np.float32(1 / 255))
    assert to_unit(u8)[255] == np.float32(255) * np.float32(1 / 255)
    with pytest.raises(TypeError):
        to_unit(u8.astype(np.int16))


def test_npz_writer_round_trip_and_cleanup(tmp_path):
    arrays = {"a": np.arange(6, dtype=np.int32).reshape(2, 3), "order": np.array(["i000", "i001"]),
              "u8": np.full((4, 4, 3), 7, np.uint8)}
    with artifacts.NpzWriter(tmp_path / "x.npz") as w:
        for k, v in arrays.items():
            w.add(k, v)
    with np.load(tmp_path / "x.npz") as z:
        assert sorted(z.files) == sorted(arrays)
        for k, v in arrays.items():
            assert np.array_equal(z[k], v) and z[k].dtype == v.dtype
    with pytest.raises(RuntimeError), artifacts.NpzWriter(tmp_path / "y.npz") as w:
        w.add("a", np.zeros(3))
        raise RuntimeError("boom")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.npz"]


def test_load_bucket_pixels(tmp_path):
    u8 = np.random.default_rng(1).integers(0, 256, (48, 96, 3), dtype=np.uint8)
    with artifacts.NpzWriter(tmp_path / "images_70.npz") as w:
        w.add("pixels_i007", u8)
        w.add("pixels_bad", u8[..., :2])
    x = load_bucket_pixels(70, "i007", s0_dir=tmp_path)
    assert x.shape == (1, 3, 48, 96) and x.dtype == np.float32 and x.flags.c_contiguous
    assert np.array_equal(x[0].transpose(1, 2, 0), u8.astype(np.float32) * np.float32(1 / 255))
    with pytest.raises(ValueError):
        load_bucket_pixels(70, "bad", s0_dir=tmp_path)


def test_layout_writer_checks_the_sequence(tmp_path):
    special = {"pad": 0, "bos": 2, "eos": 1, "boi": 10, "image": 11, "eoi": 12}
    good = {"f": {"soft_tokens": 2, "input_ids": [2, 10, 11, 11, 12, 1]}}
    assert artifacts.write_layout(tmp_path, special, good, {})["matches_bos_boi_image_eoi_eos"]
    bad = {"f": {"soft_tokens": 2, "input_ids": [10, 11, 11, 12]}}
    out = artifacts.write_layout(tmp_path, special, bad, {})
    assert not out["matches_bos_boi_image_eoi_eos"]
    assert json.loads((tmp_path / "image_token_layout.json").read_text())["image_sequence"].startswith("DIFFERS")


def test_manifest_changes(tmp_path):
    (tmp_path / "old").write_text("aa  images/a.jpg\nbb  images/b.jpg\ncc  texts.jsonl\n")
    (tmp_path / "new").write_text("aa  images/a.jpg\nbx  images/b.jpg\ndd  README.md\n")
    assert manifest_changes(tmp_path / "old", tmp_path / "new") == ["README.md", "images/b.jpg", "texts.jsonl"]
    assert manifest_changes(tmp_path / "missing", tmp_path / "old") == ["images/a.jpg", "images/b.jpg", "texts.jsonl"]
