# group: convert
"""s3_coreml_ane: every entry point at FP16, loaded with CPU_AND_NE on this Mac's Neural Engine, against s0.

Gates (gates.toml [s3_coreml_ane]): min_cosine_text, min_cosine_image, min_ane_fraction. Also writes the activation
probe of the FP32 rewrite (probe.json) and latency. See stages/s3_coreml_ane/README.md.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import time

import coremltools as ct
import numpy as np
import torch

from egcm import probe
from egcm.buckets import BUCKETS
from egcm.convert import build_package, compile_package, contract_problems, convert, disk_free_gb, size_mb
from egcm.evaluate import Package, latency_ms, run_images, run_texts, text_feed
from egcm.fixtures import S0_DIR, load_bucket_pixels
from egcm.functions import FUNCTIONS, TEXT_FUNCTIONS, Towers, load_towers, module_for
from egcm.paths import stage_dir
from egcm.placement import ane_placement
from egcm.report import Report
from egcm.stage import run_stage

STAGE = "s3_coreml_ane"
UNITS = ct.ComputeUnit.CPU_AND_NE
PROBE_IMAGES_PER_BUCKET = 2  # the probe is diagnostic; all 200 texts, 2 images per bucket and budget
KEEP = os.environ.get("EGCM_KEEP_PACKAGES") == "1"
LATENCY = ("embed_text_16", "embed_image_70_square")


@torch.inference_mode()
def run_probe(t: Towers, log) -> dict:
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
        log(f"probe {b.function_name}")
    return probe.report(stores)


def fill(report: Report) -> None:
    t0 = time.monotonic()

    def log(msg: str) -> None:
        print(f"[s3 {time.monotonic() - t0:7.1f}s] {msg}", flush=True)

    out = stage_dir(STAGE)
    pkg, compiled = out / "EmbeddingGemma2.mlpackage", out / "EmbeddingGemma2.mlmodelc"
    log(f"disk free {disk_free_gb(out):.1f} GB")
    held = {"towers": load_towers()}

    pr = run_probe(held["towers"], log)
    (out / "probe.json").write_text(json.dumps(pr, indent=1) + "\n")
    report.details["probe"] = {k: pr[k] for k in ("sites", "max_by_kind", "top", "overflow_risk")}
    log(f"probe: {pr['sites']} sites, {len(pr['overflow_risk'])} above {probe.FP16_RISK:.0f}, "
        f"largest {pr['top'][0]}")

    def make(name: str):
        module, example = module_for(name, held["towers"])
        return convert(name, module, example, ct.precision.FLOAT16)

    sizes = build_package(FUNCTIONS, make, pkg, out / "parts", log=log)
    pad_id = held.pop("towers").pad_id
    gc.collect()
    report.details["sizes_mb"] = sizes
    report.details["contract_problems"] = problems = contract_problems(pkg, FUNCTIONS)
    if problems:
        raise AssertionError(f"I/O contract mismatch: {problems}")
    compile_package(pkg, compiled)
    report.details["compiled_mb"] = size_mb(compiled)

    placement = ane_placement(compiled, UNITS)
    report.metrics["min_ane_fraction"] = min(p["fraction"] for p in placement.values())
    report.details["placement"] = placement
    log(f"Neural Engine fraction min {report.metrics['min_ane_fraction']:.4f}; off-ANE ops: "
        f"{ {k: v['off_ane'] for k, v in placement.items() if v['off_ane']} }")

    p = Package(compiled, UNITS)
    texts = run_texts(p, pad_id)
    report.metrics["min_cosine_text"] = float(texts["cos"].min())
    report.details["text"] = {"by_function": texts["by_function"]}
    log(f"texts: min cosine {texts['cos'].min():.6f}")
    images = run_images(p, log=log)
    report.metrics["min_cosine_image"] = min(float(images[b]["cos"].min()) for b in (70, 280))
    report.details["image"] = {"by_function": images["by_function"]}

    feeds = {"embed_text_16": text_feed(np.array([2, 8071, 236787, 1], np.int32), pad_id)[1],
             "embed_image_70_square": {"pixels": np.random.default_rng(0).random((1, 3, 384, 384), np.float32)}}
    report.details["latency_cpu_and_ne"] = {fn: latency_ms(p, fn, feeds[fn]) for fn in LATENCY}
    log(f"latency: {report.details['latency_cpu_and_ne']}")
    del p
    if not KEEP:
        shutil.rmtree(compiled, ignore_errors=True)
        shutil.rmtree(pkg, ignore_errors=True)
    report.details |= {"compute_units": "CPU_AND_NE", "kept_packages": KEEP, "coremltools": ct.__version__,
                       "seconds": round(time.monotonic() - t0, 1)}
    log(f"done: text {report.metrics['min_cosine_text']:.6f}, image {report.metrics['min_cosine_image']:.6f}, "
        f"ANE {report.metrics['min_ane_fraction']:.4f}")


def main() -> None:
    run_stage(STAGE, fill)


if __name__ == "__main__":
    main()
