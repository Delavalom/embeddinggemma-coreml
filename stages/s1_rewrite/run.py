# group: convert
"""s1_rewrite: our plain-PyTorch rewrite in FP32 on CPU, loaded strictly from Google's weights, against s0's vectors.

Gates (gates.toml [s1_rewrite]): strict_load, min_cosine_text, min_cosine_image. Diagnostics in `details`:
per-length and per-bucket cosines with the worst fixtures, and the debug.npz intermediate diffs.
See stages/s1_rewrite/README.md.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from egcm.config import HF_SNAPSHOT, load
from egcm.fixtures import S0_DIR
from egcm.report import Report
from egcm.stage import run_stage
from egcm.weights import (
    build_text_tower,
    build_vision_encoder,
    check_logit_bound,
    load_hf,
    mapping_for_keys,
    text_table,
)
from stages.s1_rewrite import debug, parity

STAGE = "s1_rewrite"
THREADS = 6  # this Mac is shared; s1 is not checked for determinism


def fill(report: Report) -> None:
    t0 = time.monotonic()

    def log(msg: str) -> None:
        print(f"[s1 {time.monotonic() - t0:7.1f}s] {msg}", flush=True)

    torch.set_num_threads(THREADS)
    tcfg, vcfg, _ = load(HF_SNAPSHOT)
    keys = json.loads((S0_DIR / "keys.json").read_text())
    used, unused = mapping_for_keys(keys)
    non_audio = [k for k in unused if k.split(".")[0] not in ("audio_tower", "embed_audio")]
    report.details["keys"] = {"total": len(keys), "used": len(used), "unused_audio": len(unused) - len(non_audio),
                              "unused_other": non_audio}

    sd = load_hf(HF_SNAPSHOT)
    try:
        tower = build_text_tower(tcfg, sd)
        encoder = build_vision_encoder(vcfg, sd, tcfg.hidden)
    except RuntimeError as e:  # load_state_dict(strict=True): a missing or unexpected key
        report.metrics["strict_load"] = False
        report.details["strict_load_error"] = str(e)
        return
    report.metrics["strict_load"] = not non_audio
    table = text_table(sd, tcfg.hidden)
    del sd
    bounds = {"text": check_logit_bound(tower), "vision": check_logit_bound(encoder)}
    report.details["logit_bound_max"] = {k: max(v.values()) for k, v in bounds.items()}
    log(f"loaded {len(used)} keys strictly; logit bound max {report.details['logit_bound_max']}")

    layout = json.loads((S0_DIR / "image_token_layout.json").read_text())
    with np.load(S0_DIR / "debug.npz") as dbg:
        report.details["debug"] = {"text": debug.compare_text(tower, table, dbg),
                                   "image": debug.compare_image(encoder, tower, table, layout, dbg)}
    worst = {side: max((v["rel"], k) for k, v in d.items() if isinstance(v, dict))
             for side, d in report.details["debug"].items()}
    log(f"debug intermediates, largest relative diff: {worst}")

    texts = parity.run_texts(tower, table, tcfg, S0_DIR)
    report.metrics["min_cosine_text"] = texts["min"]
    report.details["text"] = texts
    log(f"texts: min cosine {texts['min']:.7f}")

    images = parity.run_images(encoder, tower, table, layout, S0_DIR, log)
    report.metrics["min_cosine_image"] = images["min"]
    report.details["image"] = images
    report.details |= {"torch": torch.__version__, "torch_threads": THREADS,
                       "seconds": round(time.monotonic() - t0, 1)}
    log(f"done: text {texts['min']:.7f}, image {images['min']:.7f}")


def main() -> None:
    run_stage(STAGE, fill)


if __name__ == "__main__":
    main()
