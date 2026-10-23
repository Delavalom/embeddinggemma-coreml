"""A tiny EmbeddingGemma 2 for tests: upstream reference modules with random weights, their Hugging Face-named state
dict, and the rewrite built from that state dict through egcm.weights (the real mapping, strict).

Shapes keep every structural feature of the real model: GQA on sliding layers, one kv head and a wider head on
full layers, layer-major PLE, axial vision RoPE, 3x3 pooling. Only the sizes are small.
"""

import torch
import upstream_towers as ut

from egcm.buckets import Bucket
from egcm.config import FrozenMap, TextCfg, VisionCfg
from egcm.model.embedder import ImageEmbedder, TextEmbedder
from egcm.weights import build_text_tower, build_vision_encoder, build_vision_tower, text_table

TEXT = TextCfg(
    hidden=32, layers=3, heads=4, ffn=64, ple_dim=16, out_dim=24, vocab=40, eps=1e-6,
    layer_types=("sliding_attention", "full_attention", "sliding_attention"),
    head_dim=FrozenMap({"sliding_attention": 8, "full_attention": 16}),
    kv_heads=FrozenMap({"sliding_attention": 2, "full_attention": 1}),
    theta=FrozenMap({"sliding_attention": 1e4, "full_attention": 1e6}),
    pad_id=0, sliding_window=512,
)
VISION = VisionCfg(hidden=48, layers=2, heads=3, kv_heads=3, head_dim=16, ffn=96, patch=16, pool=3, theta=100.0,
                   eps=1e-6, pos_size=16)
BUCKET = Bucket(0, "tiny", 96, 144)  # grid 6 x 9 = 54 patches, 2 x 3 = 6 soft tokens
PREFIX, SUFFIX = [2, 5], [7, 1]  # stand-ins for [bos, <|image>] and [<image|>, eos]


def reference(seed: int = 0):
    """(text model, vision model, embed_vision) with random weights; norm weights in [0.5, 1.5]."""
    torch.manual_seed(seed)
    t = TEXT
    text = ut.TextModel(ut.Cfg(t.hidden, t.heads, t.ffn, t.eps), t.vocab, t.out_dim, t.ple_dim, t.layer_types,
                        dict(t.head_dim), dict(t.kv_heads), dict(t.theta))
    v = VISION
    vision = ut.VisionModel(ut.Cfg(v.hidden, v.heads, v.ffn, v.eps), v.head_dim, v.layers, v.patch, v.pool,
                            v.pos_size, v.theta)
    mm = ut.MultimodalEmbedder(v.hidden, t.hidden, v.eps)
    for mod in (text, vision, mm):
        for name, p in [*mod.named_parameters(), *mod.named_buffers()]:
            if name.endswith(("norm.weight", "layernorm.weight", "layer_scalar")):
                p.data = torch.rand_like(p) + 0.5
            else:
                p.data = torch.randn_like(p) * (0.5 if "position_embedding" in name else 0.15)
    return text.eval(), vision.eval(), mm.eval()


def hf_state_dict(text, vision, mm) -> dict[str, torch.Tensor]:
    return ({f"language_model.{k}": v for k, v in text.state_dict().items()}
            | {f"vision_tower.{k}": v for k, v in vision.state_dict().items()}
            | {f"embed_vision.{k}": v for k, v in mm.state_dict().items()})


def rewrite(sd: dict[str, torch.Tensor], text_len: int = 10):
    """(TextEmbedder for `text_len` tokens, ImageEmbedder for BUCKET, scaled table) built through egcm.weights."""
    tower = build_text_tower(TEXT, sd)
    table = text_table(sd, TEXT.hidden)
    enc = build_vision_encoder(VISION, sd, TEXT.hidden)
    vision = build_vision_tower(enc, BUCKET)

    def embed(ids):
        return table[torch.tensor(ids)].T[None, :, None, :].contiguous()

    image = ImageEmbedder(vision, tower, embed(PREFIX), embed(SUFFIX)).eval()
    return TextEmbedder(tower, table, text_len).eval(), image, table
