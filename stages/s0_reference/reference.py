"""The official model, run the official way: sentence-transformers `encode` on CPU in FP32.

Also the manual path (processor -> EmbeddingGemma2Model.forward -> mean pool -> L2 normalize) that `encode`
is checked against, the bucket resize, and forward hooks for the debug intermediates.
"""

from __future__ import annotations

import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from sentence_transformers import SentenceTransformer

from egcm.buckets import POOL, Bucket

THREADS = 4  # fixed, so reruns are bitwise identical; small because this Mac is shared


def load(path: Path) -> SentenceTransformer:
    torch.set_num_threads(THREADS)
    m = SentenceTransformer(str(path), device="cpu", model_kwargs={"torch_dtype": torch.float32},
                            config_kwargs={"audio_config": None})
    m.eval()
    return m


def hf_model(m: SentenceTransformer):
    """The EmbeddingGemma2Model inside the sentence-transformers wrapper."""
    return m[0].model


def tokenizer(m: SentenceTransformer):
    return m[0].processor.tokenizer


def _image_kwargs(budget: int) -> dict:
    return {"processing_kwargs": {"image": {"max_soft_tokens": budget}}}


def text_features(m: SentenceTransformer, text: str, prompt_name: str) -> dict:
    return m.preprocess([text], prompt=m.prompts[prompt_name])


def encode_text(m: SentenceTransformer, text: str, prompt_name: str) -> np.ndarray:
    return m.encode(text, prompt_name=prompt_name, normalize_embeddings=True, batch_size=1).astype(np.float32)


def image_features(m: SentenceTransformer, img: Image.Image, budget: int) -> dict:
    return m.preprocess([{"image": img}], **_image_kwargs(budget))


def encode_image(m: SentenceTransformer, img: Image.Image, budget: int) -> np.ndarray:
    """The official image embedding: `encode` with a {"image": ...} dict and the budget as max_soft_tokens."""
    return m.encode({"image": img}, normalize_embeddings=True, batch_size=1,
                    **_image_kwargs(budget)).astype(np.float32)


@torch.inference_mode()
def manual_embedding(m: SentenceTransformer, features: dict) -> np.ndarray:
    """processor features -> EmbeddingGemma2Model.forward -> mean of last_hidden_state over attention_mask -> L2."""
    names = ("input_ids", "attention_mask", "pixel_values", "image_position_ids")
    out = hf_model(m)(**{k: features[k] for k in names if k in features})
    mask = features["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
    pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
    return torch.nn.functional.normalize(pooled, dim=-1)[0].numpy().astype(np.float32)


def resize_to_bucket(img: Image.Image, bucket: Bucket) -> tuple[Image.Image, torch.Tensor]:
    """Resize the uint8 image to the bucket size the way the processor resizes (torchvision bicubic, antialias).

    Returns the PIL image (lossless, uint8) to feed the processor, and the uint8 CHW tensor."""
    t = tvF.pil_to_tensor(img)
    r = tvF.resize(t, [bucket.height, bucket.width], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    return tvF.to_pil_image(r), r


def valid_patches(features: dict) -> tuple[np.ndarray, np.ndarray]:
    """The non-padding patches and their (x, y) positions (padding rows have position (-1, -1))."""
    pos = features["image_position_ids"][0]
    keep = (pos != -1).all(-1)
    return features["pixel_values"][0][keep].numpy().astype(np.float32), pos[keep].numpy().astype(np.int32)


def patches_of(chw_uint8: torch.Tensor, patch: int = 16) -> np.ndarray:
    """Patchify like `convert_image_to_patches` upstream: (py, px, c) order inside each patch, rows of patches."""
    x = chw_uint8.float() / 255.0
    c, h, w = x.shape
    x = x.reshape(c, h // patch, patch, w // patch, patch).permute(1, 3, 2, 4, 0)
    return x.reshape((h // patch) * (w // patch), -1).numpy()


def soft_token_count(features: dict) -> int:
    return int(len(valid_patches(features)[1]) // POOL**2)


@contextmanager
def capture(modules: dict[str, torch.nn.Module], inputs: dict[str, torch.nn.Module] | None = None):
    """Record the output (or, for `inputs`, the first positional input) of each module as float32 numpy."""
    store: dict[str, np.ndarray] = {}
    handles = []

    def first(x):
        return x[0] if isinstance(x, tuple | list) else x

    for name, mod in modules.items():
        handles.append(mod.register_forward_hook(
            lambda _m, _i, out, name=name: store.__setitem__(name, first(out).detach().float().numpy().copy())))
    for name, mod in (inputs or {}).items():
        handles.append(mod.register_forward_pre_hook(
            lambda _m, args, name=name: store.__setitem__(name, args[0].detach().float().numpy().copy())))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def debug_hooks(m: SentenceTransformer) -> tuple[dict, dict]:
    """(output hooks, input hooks) for the intermediates s1 compares against."""
    hf = hf_model(m)
    lm, vt = hf.language_model, hf.vision_tower
    outs = {
        "lm_per_layer_inputs": lm.ple,
        "lm_layer_00": lm.layers[0], "lm_layer_05": lm.layers[5], "lm_layer_23": lm.layers[23],
        "lm_final_norm": lm.norm, "lm_last_hidden_state": lm.embedding_projection,
        "vision_patch_embed": vt.patch_embedder, "vision_layer_00": vt.encoder.layers[0],
        "vision_layer_15": vt.encoder.layers[15], "vision_pooler": vt.pooler, "embed_vision": hf.embed_vision,
    }
    return outs, {"lm_inputs_embeds": lm.ple}


class NpzWriter:
    """Writes an .npz one array at a time, so large outputs never sit in memory together. np.load reads it."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.zip = zipfile.ZipFile(self.tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True)

    def add(self, name: str, arr) -> None:
        with self.zip.open(f"{name}.npy", "w", force_zip64=True) as f:
            np.lib.format.write_array(f, np.asanyarray(arr), allow_pickle=False)

    def close(self) -> None:
        self.zip.close()
        self.tmp.replace(self.path)
