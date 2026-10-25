# group: convert
"""s3_coreml_ane: every entry point at FP16, loaded with CPU_AND_NE on this Mac's Neural Engine, against s0.

Gates (gates.toml [s3_coreml_ane], ADR-0006): min_cosine_text, min_cosine_image (floor) and min_p1_cosine_image,
min_recall_at_10 and min_p5_spearman_top50 against the bucketed FP reference (task level), min_ane_fraction. Also
writes the activation probe of the FP32 rewrite (probe.json) and latency. See stages/s3_coreml_ane/README.md.
"""

from __future__ import annotations

import os
import shutil
import time

import coremltools as ct
import numpy as np

from egcm import parts, probe
from egcm.config import HF_SNAPSHOT, load
from egcm.convert import compile_package, contract_problems, disk_free_gb, size_mb
from egcm.e5cache import PurgingPackage
from egcm.evaluate import latency_ms, run_images, run_texts, text_feed
from egcm.functions import FUNCTIONS
from egcm.paths import stage_dir
from egcm.placement import ane_placement_per_function
from egcm.ranking import metrics as task_metrics
from egcm.report import Report
from egcm.stage import run_stage
from stages.s3_coreml_ane import probe_main

STAGE = "s3_coreml_ane"
UNITS = ct.ComputeUnit.CPU_AND_NE
KEEP = os.environ.get("EGCM_KEEP_PACKAGES") == "1"
LATENCY = ("embed_text_16", "embed_image_70_square")
# egcm.ranking metrics gated here (ADR-0006); min_ane_fraction comes from the placement
GATED = ("min_cosine_text", "min_cosine_image", "min_p1_cosine_image", "min_recall_at_10", "min_p5_spearman_top50")


def fill(report: Report) -> None:
    t0 = time.monotonic()

    def log(msg: str) -> None:
        print(f"[s3 {time.monotonic() - t0:7.1f}s] {msg}", flush=True)

    out = stage_dir(STAGE)
    pkg, compiled = out / "EmbeddingGemma2.mlpackage", out / "EmbeddingGemma2.mlmodelc"
    log(f"disk free {disk_free_gb(out):.1f} GB")
    pr = probe_main.run_in_subprocess(out / "probe.json")  # the FP32 torch model never shares a process
    report.details["probe"] = {k: pr[k] for k in ("sites", "max_by_kind", "top", "overflow_risk")}
    log(f"probe: {pr['sites']} sites, {len(pr['overflow_risk'])} above {probe.FP16_RISK:.0f}, "
        f"largest {pr['top'][0]}")

    sizes = parts.build(STAGE, FUNCTIONS, "fp16", out, log=log)
    pad_id = load(HF_SNAPSHOT)[0].pad_id
    report.details["sizes_mb"] = sizes
    report.details["contract_problems"] = problems = contract_problems(pkg, FUNCTIONS)
    if problems:
        raise AssertionError(f"I/O contract mismatch: {problems}")
    compile_package(pkg, compiled)
    report.details["compiled_mb"] = size_mb(compiled)

    placement = ane_placement_per_function(pkg, FUNCTIONS, out / "placement", UNITS, log=log)
    report.metrics["min_ane_fraction"] = min(p["fraction"] for p in placement.values())
    report.details["placement"] = placement
    log(f"Neural Engine fraction min {report.metrics['min_ane_fraction']:.4f}; off-ANE ops: "
        f"{ {k: v['off_ane'] for k, v in placement.items() if v['off_ane']} }")

    p = PurgingPackage(compiled, UNITS)
    texts = run_texts(p, pad_id)
    report.details["text"] = {"by_function": texts["by_function"]}
    log(f"texts: min cosine {texts['cos'].min():.6f}")
    images = run_images(p, log=log)
    report.details["image"] = {"by_function": images["by_function"]}
    tm = task_metrics(texts, images)
    report.metrics |= {k: tm[k] for k in GATED}
    report.details["ranking"] = {k: tm[k] for k in ("vs_bucket", "vs_raw")}
    log(f"images: min cosine {tm['min_cosine_image']:.6f}, 1st percentile {tm['min_p1_cosine_image']:.6f}; vs the "
        f"bucketed FP reference recall@10 {tm['min_recall_at_10']:.4f}, p5 spearman {tm['min_p5_spearman_top50']:.4f}")

    feeds = {"embed_text_16": text_feed(np.array([2, 8071, 236787, 1], np.int32), pad_id)[1],
             "embed_image_70_square": {"pixels": np.random.default_rng(0).random((1, 3, 384, 384), np.float32)}}
    report.details["latency_cpu_and_ne"] = {fn: latency_ms(p, fn, feeds[fn]) for fn in LATENCY}
    log(f"latency: {report.details['latency_cpu_and_ne']}")
    p.close()
    del p
    if not KEEP:
        shutil.rmtree(compiled, ignore_errors=True)
        shutil.rmtree(pkg, ignore_errors=True)
    report.details |= {"compute_units": "CPU_AND_NE", "kept_packages": KEEP, "coremltools": ct.__version__,
                       "seconds": round(time.monotonic() - t0, 1)}
    log(f"done: {report.metrics}")


def main() -> None:
    run_stage(STAGE, fill)


if __name__ == "__main__":
    main()
