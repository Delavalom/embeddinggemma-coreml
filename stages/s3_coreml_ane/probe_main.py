"""s3's activation probe of the FP32 rewrite, run in its own process (`python -m stages.s3_coreml_ane.probe_main OUT`)
so the FP32 torch weights never share memory with the conversion or the Core ML runs. Diagnostic only.

All 200 texts and the first PROBE_IMAGES_PER_BUCKET images of every bucket at each budget.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROBE_IMAGES_PER_BUCKET = 2


def run(out: Path) -> dict:
    import numpy as np
    import torch

    from egcm import probe
    from egcm.buckets import BUCKETS
    from egcm.evaluate import text_feed
    from egcm.fixtures import S0_DIR, load_bucket_pixels
    from egcm.functions import TEXT_FUNCTIONS, load_towers, module_for
    from egcm.parts import Guard

    guard = Guard()
    torch.set_grad_enabled(False)
    t = load_towers()
    stores = []
    with np.load(S0_DIR / "texts.npz") as z:
        feeds = [text_feed(z[f"ids_{k}"], t.pad_id) for k in z["order"]]
    for fn in TEXT_FUNCTIONS:
        group = [f for name, f in feeds if name == fn]
        if not group:
            continue
        module, _ = module_for(fn, t)
        with probe.probe(module, "text/") as s:
            for f in group:
                module(torch.from_numpy(f["input_ids"]), torch.from_numpy(f["attention_mask"]))
        stores.append(s)
    for b in BUCKETS:
        with np.load(S0_DIR / f"images_{b.budget}.npz") as z:
            ids = [str(o) for o, n in zip(z["order"], z["bucket"], strict=True) if str(n) == b.name]
        if not ids:
            continue
        module, _ = module_for(b.function_name, t)
        with probe.probe(module, "image/") as s:
            for i in ids[:PROBE_IMAGES_PER_BUCKET]:
                module(torch.from_numpy(load_bucket_pixels(b.budget, i)))
        stores.append(s)
        print(f"probe {b.function_name}", flush=True)
    r = probe.report(stores)
    r["images_per_bucket"] = PROBE_IMAGES_PER_BUCKET
    r["peak_footprint_gb"] = round(guard.peak, 2)
    out.write_text(json.dumps(r, indent=1) + "\n")
    return r


def run_in_subprocess(out: Path) -> dict:
    from egcm.paths import ROOT

    env = os.environ | {"PYTHONPATH": str(ROOT)}
    p = subprocess.run([sys.executable, "-m", "stages.s3_coreml_ane.probe_main", str(out)], env=env, cwd=ROOT,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"probe failed (exit {p.returncode}): {p.stderr[-2500:]}")
    return json.loads(out.read_text())


if __name__ == "__main__":
    run(Path(sys.argv[1]))
