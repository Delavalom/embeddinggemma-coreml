# group: convert
"""s2_coreml_fp32: every entry point converted at FP32 into one multifunction package, run on the CPU, against s0.

Gates (gates.toml [s2_coreml_fp32]): min_cosine_text, min_cosine_image. See stages/s2_coreml_fp32/README.md.
"""

from __future__ import annotations

import os
import shutil
import time

import coremltools as ct
import torch

from egcm import parts
from egcm.config import HF_SNAPSHOT, load
from egcm.convert import compile_package, contract_problems, disk_free_gb, size_mb
from egcm.evaluate import Package, run_images, run_texts
from egcm.functions import FUNCTIONS
from egcm.paths import stage_dir
from egcm.report import Report
from egcm.stage import run_stage

STAGE = "s2_coreml_fp32"
KEEP = os.environ.get("EGCM_KEEP_PACKAGES") == "1"  # the FP32 package is not used downstream; deleted by default


def fill(report: Report) -> None:
    t0 = time.monotonic()

    def log(msg: str) -> None:
        print(f"[s2 {time.monotonic() - t0:7.1f}s] {msg}", flush=True)

    out = stage_dir(STAGE)
    pkg, compiled = out / "EmbeddingGemma2.mlpackage", out / "EmbeddingGemma2.mlmodelc"
    log(f"disk free {disk_free_gb(out):.1f} GB")
    sizes = parts.build(STAGE, FUNCTIONS, "fp32", out, log=log)  # one process per part and per merge
    pad_id = load(HF_SNAPSHOT)[0].pad_id
    report.details["sizes_mb"] = sizes
    report.details["contract_problems"] = problems = contract_problems(pkg, FUNCTIONS)
    if problems:
        raise AssertionError(f"I/O contract mismatch: {problems}")
    compile_package(pkg, compiled)
    report.details["compiled_mb"] = size_mb(compiled)
    peak = max(sizes["peak_footprint_gb"].values())
    log(f"package {sizes['packages']['EmbeddingGemma2']['mb']:.0f} MB (parts {sum(sizes['part_mb'].values()):.0f} "
        f"MB); peak footprint per process {peak} GB; disk free {disk_free_gb(out):.1f} GB")

    p = Package(compiled, ct.ComputeUnit.CPU_ONLY)
    texts = run_texts(p, pad_id)
    report.metrics["min_cosine_text"] = float(texts["cos"].min())
    report.details["text"] = {"by_function": texts["by_function"]}
    log(f"texts: min cosine {texts['cos'].min():.7f}")
    images = run_images(p, log=log)
    report.metrics["min_cosine_image"] = min(float(images[b]["cos"].min()) for b in (70, 280))
    report.details["image"] = {"by_function": images["by_function"]}
    del p
    if not KEEP:
        shutil.rmtree(compiled, ignore_errors=True)
        shutil.rmtree(pkg, ignore_errors=True)
    report.details |= {"kept_packages": KEEP, "torch": torch.__version__, "coremltools": ct.__version__,
                       "seconds": round(time.monotonic() - t0, 1)}
    log(f"done: text {report.metrics['min_cosine_text']:.7f}, image {report.metrics['min_cosine_image']:.7f}")


def main() -> None:
    run_stage(STAGE, fill)


if __name__ == "__main__":
    main()
