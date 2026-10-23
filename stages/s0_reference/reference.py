"""The official model, run the official way: sentence-transformers `encode` on CPU in FP32.

Also the manual path (processor -> EmbeddingGemma2Model.forward -> mean pool -> L2 normalize) that `encode`
is checked against, the bucket resize, and forward hooks for the debug intermediates.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from sentence_transformers import SentenceTransformer

from egcm.buckets import Bucket

THREADS = 4  # fixed, so reruns are bitwise identical; small because this Mac is shared
PROMPT_FOR_KIND = {"query": "query", "document": "document"}  # fixture kind -> sentence-transformers prompt name


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


def prompt_name(kind: str) -> str:
    """Strict: an unknown fixture kind is an error, never a silent default."""
    return PROMPT_FOR_KIND[kind]


def _image_kwargs(budget: int) -> dict:
    return {"processing_kwargs": {"image": {"max_soft_tokens": budget}}}


def text_features(m: SentenceTransformer, text: str, name: str) -> dict:
    return m.preprocess([text], prompt=m.prompts[name])


def encode_text(m: SentenceTransformer, text: str, name: str) -> np.ndarray:
    m.prompts[name]  # strict: KeyError on an unknown prompt name
    return m.encode(text, prompt_name=name, normalize_embeddings=True, batch_size=1).astype(np.float32)


def image_features(m: SentenceTransformer, img: Image.Image, budget: int) -> dict:
    return m.preprocess([{"image": img}], **_image_kwargs(budget))


def processor_soft_tokens(m: SentenceTransformer, img: Image.Image, budget: int) -> int:
    """The image processor's own `num_soft_tokens_per_image` for this image and budget."""
    out = m[0].processor.image_processor(images=[img], max_soft_tokens=budget)
    return int(out["num_soft_tokens_per_image"][0])


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


def resize_to_bucket(img: Image.Image, bucket: Bucket) -> tuple[Image.Image, np.ndarray]:
    """Resize the uint8 image to the bucket size the way the processor resizes (torchvision bicubic, antialias).

    Returns the PIL image (lossless, uint8) to feed the processor, and the same pixels as uint8 HxWx3."""
    t = tvF.pil_to_tensor(img)
    r = tvF.resize(t, [bucket.height, bucket.width], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    return tvF.to_pil_image(r), r.permute(1, 2, 0).contiguous().numpy()


def valid_patches(features: dict) -> tuple[np.ndarray, np.ndarray]:
    """The processor's non-padding patches (float32) and their (x, y) positions (padding rows are (-1, -1))."""
    pos = features["image_position_ids"][0]
    keep = (pos != -1).all(-1)
    return features["pixel_values"][0][keep].numpy(), pos[keep].numpy().astype(np.int32)


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
