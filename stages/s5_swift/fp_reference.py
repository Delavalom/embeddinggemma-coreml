"""The FP reference on Swift's exact pixels (ADR-0008): one budget per process, the s1 rewrite in FP32 on the CPU.

    python -m stages.s5_swift.fp_reference --budget 70 --parity <parity.json> --pixels <dir> --out <npz> [--s0 <dir>]

The s1 rewrite is s0's model rebuilt in plain PyTorch, loaded strictly from the same weights; s1 gates it at cosine
>= 0.9999 against s0 and measures 1 - 1e-8 on every fixture image. Images whose Swift bucket image is byte-identical
to s0's (same sha256; the PNG fixtures) already have their FP reference: s0's `emb_bucket`. Only the others (the
JPEGs, decoded by ImageIO instead of PIL) are embedded here. Writes `<id>` -> vector for those, and `peak_rss_bytes`.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from egcm.buckets import BUCKETS
from egcm.config import HF_SNAPSHOT, load
from egcm.fixtures import S0_DIR
from egcm.functions import image_embedder
from egcm.weights import build_text_tower, build_vision_encoder, load_hf, text_table
from stages.s5_swift.compare import sha256_hwc
from stages.s5_swift.pixels import model_input, swift_pixels

THREADS = 6  # as s1: this Mac is shared


def needed(parity: dict, s0: Path, budget: int) -> list[str]:
    """Image ids at `budget` whose Swift bucket image differs from s0's stored one (sorted)."""
    with np.load(s0 / f"images_{budget}.npz") as z:
        return sorted(k.rsplit("_", 1)[0] for k, e in parity["images"].items()
                      if int(e["budget"]) == budget and e["sha256"] != sha256_hwc(z[f"pixels_{k.rsplit('_', 1)[0]}"]))


@torch.inference_mode()
def run(parity: dict, pixels: Path, budget: int, ids: list[str], s0: Path, log=print) -> dict[str, np.ndarray]:
    t0 = time.monotonic()
    torch.set_num_threads(THREADS)
    tcfg, vcfg, _ = load(HF_SNAPSHOT)
    sd = load_hf(HF_SNAPSHOT)
    tower, encoder = build_text_tower(tcfg, sd), build_vision_encoder(vcfg, sd, tcfg.hidden)
    table = text_table(sd, tcfg.hidden)
    del sd
    layout = json.loads((s0 / "image_token_layout.json").read_text())
    out: dict[str, np.ndarray] = {}
    for bucket in (b for b in BUCKETS if b.budget == budget):
        mine = [i for i in ids if parity["images"][f"{i}_{budget}"]["bucket"] == bucket.name]
        if not mine:
            continue
        emb = image_embedder(encoder, tower, table, layout, bucket)
        for i in mine:
            key = f"{i}_{budget}"
            out[i] = emb(torch.from_numpy(model_input(swift_pixels(parity["images"][key], pixels, key))))[0].numpy()
        log(f"[s5 fp {budget} {time.monotonic() - t0:6.0f}s] {bucket.function_name}: {len(mine)}")
        del emb
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m stages.s5_swift.fp_reference")
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--parity", type=Path, required=True)
    ap.add_argument("--pixels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--s0", type=Path, default=S0_DIR)
    a = ap.parse_args(argv)
    parity = json.loads(a.parity.read_text())
    ids = needed(parity, a.s0, a.budget)
    out = run(parity, a.pixels, a.budget, ids, a.s0, log=lambda s: print(s, file=sys.stderr, flush=True)) if ids else {}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    np.savez(a.out, peak_rss_bytes=np.int64(rss), **out)


if __name__ == "__main__":
    main()
