"""The spec section 7 entry points: their names, the fixed I/O contract, and the torch module behind each.

Contract (mirrored by the Swift runtime and stages/s5_swift/make_contract_model.py on feat/swift; keep identical):
- `embed_text_{16,32,64,128}`: `input_ids` int32 [1, L], `attention_mask` int32 [1, L] (1 = token, 0 = padding).
- `embed_image_{70,280}_{tall,portrait,square,landscape,wide}`: `pixels` fp32 [1, 3, H, W] in [0, 1].
- Every function outputs `embedding` fp32 [1, 768], L2-normalized.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from egcm.buckets import BUCKETS, Bucket
from egcm.config import HF_SNAPSHOT, load
from egcm.fixtures import S0_DIR
from egcm.model.embedder import TEXT_LENGTHS, ImageEmbedder, TextEmbedder
from egcm.model.text import TextTower
from egcm.model.vision import VisionEncoder
from egcm.weights import build_text_tower, build_vision_encoder, build_vision_tower, load_hf, text_table

TEXT_FUNCTIONS = tuple(f"embed_text_{n}" for n in TEXT_LENGTHS)
IMAGE_FUNCTIONS = tuple(b.function_name for b in BUCKETS)
FUNCTIONS = TEXT_FUNCTIONS + IMAGE_FUNCTIONS
DEFAULT_FUNCTION = "embed_text_16"
EMBEDDING_DIM = 768


def text_length(name: str) -> int:
    return int(name.removeprefix("embed_text_"))


def bucket_for(name: str) -> Bucket:
    return next(b for b in BUCKETS if b.function_name == name)


class TextFunction(nn.Module):
    """TextEmbedder with the contract's int32 attention mask (cast to float inside the model)."""

    def __init__(self, embedder: TextEmbedder):
        super().__init__()
        self.embedder = embedder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.embedder(input_ids, attention_mask.to(torch.float32))


def image_affixes(layout: dict, bucket: Bucket) -> tuple[list[int], list[int]]:
    """The token ids before and after the image placeholders in s0's recorded sequence for this bucket."""
    entry = layout["buckets"][bucket.function_name]
    ids, image = entry["input_ids"], layout["special_token_ids"]["image"]
    where = [i for i, t in enumerate(ids) if t == image]
    if len(where) != bucket.tokens or where != list(range(where[0], where[0] + len(where))):
        raise AssertionError(f"{bucket.function_name}: {len(where)} image placeholders, expected {bucket.tokens} "
                             f"in one run")
    return ids[: where[0]], ids[where[-1] + 1:]


def embed_ids(table: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """Scaled text embeddings of fixed tokens, (1, C, 1, n)."""
    return table[torch.tensor(ids, dtype=torch.long)].T[None, :, None, :].contiguous()


def image_embedder(encoder: VisionEncoder, tower: TextTower, table: torch.Tensor, layout: dict,
                   bucket: Bucket) -> ImageEmbedder:
    prefix, suffix = image_affixes(layout, bucket)
    return ImageEmbedder(build_vision_tower(encoder, bucket), tower, embed_ids(table, prefix),
                         embed_ids(table, suffix)).eval()


@dataclass
class Towers:
    tower: TextTower
    encoder: VisionEncoder
    table: torch.Tensor
    layout: dict
    pad_id: int


def load_towers(snapshot: Path = HF_SNAPSHOT, s0: Path = S0_DIR) -> Towers:
    """The FP32 rewrite loaded strictly from the snapshot, plus s0's image token layout."""
    tcfg, vcfg, _ = load(snapshot)
    sd = load_hf(snapshot)
    towers = Towers(tower=build_text_tower(tcfg, sd), encoder=build_vision_encoder(vcfg, sd, tcfg.hidden),
                    table=text_table(sd, tcfg.hidden),
                    layout=json.loads((s0 / "image_token_layout.json").read_text()), pad_id=tcfg.pad_id)
    del sd
    return towers


def module_for(name: str, t: Towers) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    """(module, example inputs for tracing) of one entry point."""
    if name in TEXT_FUNCTIONS:
        n = text_length(name)
        return (TextFunction(TextEmbedder(t.tower, t.table, n)).eval(),
                (torch.zeros(1, n, dtype=torch.int32), torch.ones(1, n, dtype=torch.int32)))
    b = bucket_for(name)
    gen = torch.Generator().manual_seed(0)
    pixels = torch.rand(1, 3, b.height, b.width, generator=gen)
    return image_embedder(t.encoder, t.tower, t.table, t.layout, b), (pixels,)
