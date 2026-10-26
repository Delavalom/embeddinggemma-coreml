"""s4's task-level metrics (egcm.ranking) on the Swift vectors s5 wrote (artifacts/s5_swift/parity.json).

    uv run --exact --group convert python -m scripts.s5_task_metrics [--parity <parity.json>] [--s0 <dir>]

Diagnostic only (draft ADR-0008): no report, no gate. Prints the same keys s4 gates on (recall@10 and p5 spearman
top-50 against the bucketed FP reference, t2i top-1 in top-10 and the i2i label p@10 drop against exact-aspect), next
to s4's values and the `[s4_compress]` thresholds. Pure numpy; no model is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from egcm import ranking
from egcm.buckets import BUDGETS
from egcm.fixtures import S0_DIR
from egcm.gates import gates
from egcm.metrics import cosine_rows
from egcm.paths import ARTIFACTS

KEYS = ["min_cosine_text", "min_cosine_image", "min_p1_cosine_image", "min_cosine", "min_recall_at_10",
        "min_p5_spearman_top50", "min_t2i_top1_in_top10", "max_i2i_label_p10_drop"]


def swift_inputs(parity: dict, s0: Path) -> tuple[dict, dict]:
    """parity.json + s0 in the shape egcm.ranking.metrics takes (egcm.evaluate.run_texts / run_images)."""
    with np.load(s0 / "texts.npz") as z:
        order = [str(x) for x in z["order"]]
        ref = z["emb"]
    emb = np.array([parity["texts"][k]["emb"] for k in order], np.float32)
    texts = {"order": order, "emb": emb, "ref": ref, "cos": cosine_rows(emb, ref)}
    images: dict = {}
    for b in BUDGETS:
        with np.load(s0 / f"images_{b}.npz") as z:
            order = [str(x) for x in z["order"]]
            ref, raw = z["emb_bucket"], z["emb_raw"]
        emb = np.array([parity["images"][f"{k}_{b}"]["emb"] for k in order], np.float32)
        images[b] = {"order": order, "emb": emb, "ref_bucket": ref, "ref_raw": raw, "cos": cosine_rows(emb, ref)}
    return texts, images


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.s5_task_metrics")
    ap.add_argument("--parity", type=Path, default=ARTIFACTS / "s5_swift" / "parity.json")
    ap.add_argument("--s0", type=Path, default=S0_DIR)
    a = ap.parse_args()
    got = ranking.metrics(*swift_inputs(json.loads(a.parity.read_text()), a.s0))
    s4 = json.loads((ARTIFACTS / "s4_compress" / "report.json").read_text())
    s4v = s4["details"]["variants"][s4["details"]["winner"]]
    limits = gates("s4_compress")
    print(f"{'metric':26} {'swift (s5)':>11} {'s4':>11} {'s4 gate':>9}")
    for k in KEYS:
        lim = limits.get(k)
        print(f"{k:26} {got[k]:11.4f} {s4v.get(k, float('nan')):11.4f} {'' if lim is None else lim:>9}")
    print(json.dumps({"vs_bucket": got["vs_bucket"]["by_budget"], "vs_raw": got["vs_raw"]["by_budget"]},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
