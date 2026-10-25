# group: convert
"""s4_compress: compressed variants of the FP16 model; keeps the smallest that passes every s4 gate.

Gates (gates.toml [s4_compress]): min_cosine, min_recall_at_10, min_p5_spearman_top50, max_download_mb.
See stages/s4_compress/README.md.
"""

from __future__ import annotations

import json
import time

import coremltools as ct
import numpy as np

from egcm import parts
from egcm.compress import MEASURE_ONLY, RECIPES
from egcm.config import HF_SNAPSHOT, load
from egcm.convert import compile_package, contract_problems, disk_free_gb, size_mb
from egcm.e5cache import PurgingPackage
from egcm.e5prune import BundlePruner
from egcm.evaluate import latency_ms, run_images, run_texts, text_feed
from egcm.functions import FUNCTIONS
from egcm.gates import gates as stage_gates
from egcm.paths import stage_dir
from egcm.placement import ane_placement_per_function
from egcm.report import Report
from egcm.stage import run_stage
from stages.s4_compress import ranking

STAGE = "s4_compress"
UNITS = ct.ComputeUnit.CPU_AND_NE  # the shipping configuration


def passes(m: dict, gates: dict) -> dict[str, bool]:
    return {"min_cosine": m["min_cosine"] >= gates["min_cosine"],
            "min_recall_at_10": m["min_recall_at_10"] >= gates["min_recall_at_10"],
            "min_p5_spearman_top50": m["min_p5_spearman_top50"] >= gates["min_p5_spearman_top50"],
            "max_download_mb": m["download_mb"] <= gates["max_download_mb"]}


def evaluate_variant(recipe: str, desc: str, sizes: dict, out, pad_id: int, gates: dict, log,
                     placement: bool = True) -> dict:
    """Metrics for out/<recipe>.mlpackage on CPU_AND_NE. `placement=False` skips the Neural Engine placement
    analysis (about 90 minutes for 14 functions; scripts.measure_s4_eval)."""
    pkg, compiled, zipped = out / f"{recipe}.mlpackage", out / f"{recipe}.mlmodelc", out / f"{recipe}.zip"
    if problems := contract_problems(pkg, FUNCTIONS):
        raise AssertionError(f"{recipe}: I/O contract mismatch: {problems}")
    ranking.zip_package(pkg, zipped)
    compile_package(pkg, compiled)
    p = PurgingPackage(compiled, UNITS)
    texts, images = run_texts(p, pad_id), run_images(p, log=log)
    m = ranking.metrics(texts, images)
    m |= {"description": desc, "package_mb": sizes["package_mb"], "download_mb": size_mb(zipped),
          "by_function": {"text": texts["by_function"], "image": images["by_function"]}}
    m["passes"] = passes(m, gates)
    if placement:
        m |= placement_metrics(pkg, out, log)
    feeds = {"embed_text_16": text_feed(np.array([2, 8071, 236787, 1], np.int32), pad_id)[1],
             "embed_image_70_square": {"pixels": np.random.default_rng(0).random((1, 3, 384, 384), np.float32)}}
    m["latency_cpu_and_ne"] = {fn: latency_ms(p, fn, f) for fn, f in feeds.items()}
    p.close()
    del p
    ranking.remove(compiled)
    log(f"{recipe}: download {m['download_mb']:.0f} MB, min cosine {m['min_cosine']:.5f}, recall@10 "
        f"{m['min_recall_at_10']:.3f} (vs bucket {m['vs_bucket']['min_recall_at_10']:.3f}), p5 spearman "
        f"{m['min_p5_spearman_top50']:.3f} (vs bucket {m['vs_bucket']['min_p5_spearman_top50']:.3f}), "
        f"passes {m['passes']}")
    return m


def placement_metrics(pkg, out, log) -> dict:
    placement = ane_placement_per_function(pkg, FUNCTIONS, out / "placement", UNITS, log=log)
    return {"ane_fraction_min": min(v["fraction"] for v in placement.values()),
            "ane_fraction": {k: v["fraction"] for k, v in placement.items()},
            "off_ane": {k: v["off_ane"] for k, v in placement.items() if v["off_ane"]}}


def fill(report: Report) -> None:
    t0 = time.monotonic()

    def log(msg: str) -> None:
        print(f"[s4 {time.monotonic() - t0:7.1f}s] {msg}", flush=True)

    gates = stage_gates(STAGE)
    out = stage_dir(STAGE)
    log(f"disk free {disk_free_gb(out):.1f} GB")
    built = parts.build(STAGE, FUNCTIONS, "fp16", out, recipes=list(RECIPES), log=log)
    sizes = {r: {"package_mb": v["mb"]} for r, v in built["packages"].items()}
    pad_id = load(HF_SNAPSHOT)[0].pad_id
    report.details["build"] = built

    variants: dict[str, dict] = {}
    winner = None
    with BundlePruner() as pruner:  # Core ML's E5 cache grows ~1.1 GB a minute while evaluating
        for recipe, (desc, _) in RECIPES.items():  # smallest first: the first that passes everything wins
            variants[recipe] = evaluate_variant(recipe, desc, sizes[recipe], out, pad_id, gates, log)
            if all(variants[recipe]["passes"].values()):
                winner = recipe
                break
    report.details["e5_pruned_mb"] = round(pruner.deleted_mb)
    report.details["variants"] = variants
    # with no winner, report the variant closest to passing (most gates met, then smallest download)
    chosen = winner or min(variants, key=lambda r: (-sum(variants[r]["passes"].values()), variants[r]["download_mb"]))
    for recipe in RECIPES:
        if recipe != chosen:
            ranking.remove(out / f"{recipe}.mlpackage", out / f"{recipe}.zip")
    final, final_zip = out / "EmbeddingGemma2.mlpackage", out / "EmbeddingGemma2.mlpackage.zip"
    ranking.remove(final, final_zip, out / "checksums.json")
    (out / f"{chosen}.mlpackage").rename(final)
    (out / f"{chosen}.zip").unlink()
    if winner:  # the shippable package
        ranking.zip_package(final, final_zip)
        (out / "checksums.json").write_text(json.dumps(ranking.checksums(final, final_zip), indent=1) + "\n")
    m = variants[chosen]
    report.metrics |= {k: m[k] for k in ("min_cosine", "min_recall_at_10", "min_p5_spearman_top50")}
    report.metrics["max_download_mb"] = m["download_mb"]
    report.details |= {"chosen": chosen, "winner": winner, "package": final.name,
                       "seconds": round(time.monotonic() - t0, 1)}
    log(f"done: chosen {chosen} (winner {winner})")


def measure(recipes: list[str]) -> dict:
    """Build and evaluate the named variants without the stage runner (no report.json, no gate verdict): for
    measurements while an upstream stage has not passed (ADR-0007). Writes artifacts/s4_compress/measure/
    measurement.json; keeps no package."""
    t0 = time.monotonic()

    def log(msg: str) -> None:
        print(f"[s4 measure {time.monotonic() - t0:7.1f}s] {msg}", flush=True)

    gates = stage_gates(STAGE)
    out = stage_dir(STAGE) / "measure"
    out.mkdir(exist_ok=True)
    log(f"disk free {disk_free_gb(out):.1f} GB; recipes {recipes}")
    built = parts.build(STAGE, FUNCTIONS, "fp16", out, recipes=recipes, log=log)
    pad_id = load(HF_SNAPSHOT)[0].pad_id
    variants = {}
    with BundlePruner() as pruner:
        for r in recipes:
            desc, sizes = (RECIPES | MEASURE_ONLY)[r][0], {"package_mb": built["packages"][r]["mb"]}
            variants[r] = evaluate_variant(r, desc, sizes, out, pad_id, gates, log)
            ranking.remove(out / f"{r}.mlpackage", out / f"{r}.zip")
            (out / "measurement.json").write_text(json.dumps({"variants": variants, "build": built}, indent=1) + "\n")
    res = {"variants": variants, "build": built, "e5_pruned_mb": round(pruner.deleted_mb),
           "seconds": round(time.monotonic() - t0, 1)}
    (out / "measurement.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main() -> None:
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--measure":
        measure(sys.argv[2].split(","))
        return
    run_stage(STAGE, fill)


if __name__ == "__main__":
    main()
