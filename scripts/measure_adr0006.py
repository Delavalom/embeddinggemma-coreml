"""Evidence for ADR-0006: the FP16 package with image functions on the GPU and text on the Neural Engine.

    caffeinate -i -s uv run --exact --group convert python -m scripts.measure_adr0006

Builds the s3 FP16 package (egcm.parts, resumable) in artifacts/adr0006/, then:
- every fixture image with its image function loaded `CPU_AND_GPU`; cosines against s0's bucketed vectors;
- every fixture text with its text function loaded `CPU_AND_NE` from the same package;
- latency (median / p90 of 20 after 3 warm-ups) of embed_image_70_square, embed_image_280_square and embed_text_16
  on both `CPU_AND_GPU` and `CPU_AND_NE`.
Writes artifacts/adr0006/measure.json and deletes the package (EGCM_KEEP_PACKAGES=1 keeps it). Not a stage: no
report, no gate.
"""

from __future__ import annotations

import json
import os
import shutil
import time

import coremltools as ct
import numpy as np

from egcm import parts
from egcm.config import HF_SNAPSHOT, load
from egcm.convert import compile_package
from egcm.e5cache import PurgingPackage
from egcm.e5prune import BundlePruner
from egcm.evaluate import latency_ms, run_images, run_texts, text_feed
from egcm.functions import FUNCTIONS
from egcm.paths import ARTIFACTS

OUT = ARTIFACTS / "adr0006"
LATENCY = ("embed_image_70_square", "embed_image_280_square", "embed_text_16")


def main() -> None:
    t0 = time.monotonic()

    def log(msg: str) -> None:
        print(f"[adr0006 {time.monotonic() - t0:7.1f}s] {msg}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    built = parts.build("s3_coreml_ane", FUNCTIONS, "fp16", OUT, log=log)
    pkg, compiled = OUT / "EmbeddingGemma2.mlpackage", OUT / "EmbeddingGemma2.mlmodelc"
    compile_package(pkg, compiled)
    pad_id = load(HF_SNAPSHOT)[0].pad_id
    res: dict = {"package_mb": built["packages"]["EmbeddingGemma2"]["mb"]}
    with BundlePruner() as pruner:
        gpu = PurgingPackage(compiled, ct.ComputeUnit.CPU_AND_GPU)
        images = run_images(gpu, log=log)
        res["image_cpu_and_gpu"] = {"min": min(float(images[b]["cos"].min()) for b in (70, 280)),
                                    "by_function": images["by_function"]}
        gpu.close()
        ane = PurgingPackage(compiled, ct.ComputeUnit.CPU_AND_NE)
        texts = run_texts(ane, pad_id)
        res["text_cpu_and_ne"] = {"min": float(texts["cos"].min()), "by_function": texts["by_function"]}
        ane.close()
        feeds = {"embed_text_16": text_feed(np.array([2, 8071, 236787, 1], np.int32), pad_id)[1]}
        rng = np.random.default_rng(0)
        feeds["embed_image_70_square"] = {"pixels": rng.random((1, 3, 384, 384), np.float32)}
        feeds["embed_image_280_square"] = {"pixels": rng.random((1, 3, 768, 768), np.float32)}
        res["latency_ms"] = {}
        for units in ("CPU_AND_GPU", "CPU_AND_NE"):
            p = PurgingPackage(compiled, getattr(ct.ComputeUnit, units))
            res["latency_ms"][units] = {fn: latency_ms(p, fn, feeds[fn]) for fn in LATENCY}
            p.close()
            log(f"latency {units}: { {k: round(v['median_ms'], 1) for k, v in res['latency_ms'][units].items()} }")
    res |= {"e5_pruned_mb": round(pruner.deleted_mb), "seconds": round(time.monotonic() - t0, 1),
            "peak_footprint_gb": built["peak_footprint_gb"]}
    (OUT / "measure.json").write_text(json.dumps(res, indent=1) + "\n")
    log(f"images on GPU min {res['image_cpu_and_gpu']['min']:.5f}; text on ANE min {res['text_cpu_and_ne']['min']:.6f}")
    if os.environ.get("EGCM_KEEP_PACKAGES") != "1":
        shutil.rmtree(compiled, ignore_errors=True)
        shutil.rmtree(pkg, ignore_errors=True)


if __name__ == "__main__":
    main()
