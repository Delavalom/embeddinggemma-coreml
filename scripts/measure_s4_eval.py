"""s4 measurements without the per-function placement analysis, except for the variants named after --placement.

    caffeinate -i -s uv run --exact --group convert python -m scripts.measure_s4_eval \\
        vis4kv8_text8,fp16,int8_table4 --placement vis4kv8_text8

`stages.s4_compress.run --measure` spends about 90 minutes per variant on `ane_placement_per_function` (14
functions); placement doesn't change the ranking metrics, and s3 already recorded the FP16 placement. This builds (or
resumes) artifacts/s4_compress/measure/ with `egcm.parts.build`: if build.json lists the same variants in another
order, that order is kept so the build resumes instead of starting over. Then it evaluates the variants in the order
given and writes measure/fast.json after each one; the placement of the --placement variants comes last, also into
fast.json. Keeps the packages and zips. Not a stage: no report, no gate.
"""

from __future__ import annotations

import argparse
import json
import time

from egcm import parts
from egcm.compress import MEASURE_ONLY, RECIPES
from egcm.config import HF_SNAPSHOT, load
from egcm.e5exit import ExitPurge
from egcm.e5prune import BundlePruner
from egcm.functions import FUNCTIONS
from egcm.gates import gates as stage_gates
from egcm.paths import stage_dir
from stages.s4_compress.run import STAGE, evaluate_variant, placement_metrics


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.measure_s4_eval")
    ap.add_argument("variants", help="comma-separated recipe / measure-only names, evaluated in this order")
    ap.add_argument("--placement", default="", help="comma-separated variants to run the placement analysis for")
    a = ap.parse_args()
    order, with_placement = a.variants.split(","), [v for v in a.placement.split(",") if v]
    unknown = [v for v in order + with_placement if v not in RECIPES | MEASURE_ONLY]
    if unknown or not set(with_placement) <= set(order):
        raise SystemExit(f"unknown variants {unknown}, or --placement not among the variants")
    t0 = time.monotonic()

    def log(msg: str) -> None:
        print(f"[s4 eval {time.monotonic() - t0:7.1f}s] {msg}", flush=True)

    out = stage_dir(STAGE) / "measure"
    out.mkdir(parents=True, exist_ok=True)
    state = out / "build.json"
    keys = json.loads(state.read_text())["keys"] if state.exists() else []
    keys = keys if set(keys) == set(order) else order
    built = parts.build(STAGE, FUNCTIONS, "fp16", out, recipes=keys, log=log)
    gates, pad_id = stage_gates(STAGE), load(HF_SNAPSHOT)[0].pad_id
    res: dict = {"variants": {}, "build": built}

    def save() -> None:
        res["seconds"] = round(time.monotonic() - t0, 1)
        (out / "fast.json").write_text(json.dumps(res, indent=1) + "\n")

    with BundlePruner():  # Core ML's E5 cache grows ~1.1 GB a minute while evaluating
        for r in order:
            sizes = {"package_mb": built["packages"][r]["mb"]}
            res["variants"][r] = evaluate_variant(r, (RECIPES | MEASURE_ONLY)[r][0], sizes, out, pad_id, gates,
                                                  log, placement=False)
            save()
        for r in with_placement:
            res["variants"][r] |= placement_metrics(out / f"{r}.mlpackage", out, log)
            log(f"{r}: Neural Engine fraction min {res['variants'][r]['ane_fraction_min']:.5f}")
            save()


if __name__ == "__main__":
    with ExitPurge():  # delete the E5 bundles this run created, also on failure
        main()
