# group: convert
"""s4_compress: compressed variants of the FP16 package; keeps the smallest that passes every s4 gate.

Gates (gates.toml [s4_compress]): min_cosine, min_recall_at_10, min_p5_spearman_top50, max_download_mb.
See stages/s4_compress/README.md.
"""

from __future__ import annotations

import gc
import json
import shutil
import time

import coremltools as ct
import numpy as np

from egcm.compress import RECIPES
from egcm.convert import build_package, compile_package, contract_problems, convert, disk_free_gb, size_mb
from egcm.evaluate import Package, latency_ms, run_images, run_texts, text_feed
from egcm.functions import FUNCTIONS, load_towers, module_for
from egcm.gates import gates as stage_gates
from egcm.paths import stage_dir
from egcm.placement import ane_placement
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


def fill(report: Report) -> None:
    t0 = time.monotonic()

    def log(msg: str) -> None:
        print(f"[s4 {time.monotonic() - t0:7.1f}s] {msg}", flush=True)

    gates = stage_gates(STAGE)
    out = stage_dir(STAGE)
    fp16 = out / "fp16_parts"
    log(f"disk free {disk_free_gb(out):.1f} GB")
    towers = load_towers()
    pad_id = towers.pad_id
    fp16.mkdir(exist_ok=True)
    for name in FUNCTIONS:  # FP16 parts once; every recipe starts from them
        part = fp16 / f"{name}.mlpackage"
        if not part.exists():
            module, example = module_for(name, towers)
            convert(name, module, example, ct.precision.FLOAT16).save(str(part))
            log(f"fp16 {name}: {size_mb(part):.0f} MB")
    del towers
    gc.collect()

    variants: dict[str, dict] = {}
    winner = None
    for recipe, (desc, compress) in RECIPES.items():
        pkg, compiled, zipped = out / f"{recipe}.mlpackage", out / f"{recipe}.mlmodelc", out / f"{recipe}.zip"

        def make(name: str, compress=compress):
            return compress(ct.models.MLModel(str(fp16 / f"{name}.mlpackage"), skip_model_load=True))

        sizes = build_package(FUNCTIONS, make, pkg, out / "parts", log=log)
        if problems := contract_problems(pkg, FUNCTIONS):
            raise AssertionError(f"{recipe}: I/O contract mismatch: {problems}")
        ranking.zip_package(pkg, zipped)
        compile_package(pkg, compiled)
        p = Package(compiled, UNITS)
        texts, images = run_texts(p, pad_id), run_images(p, log=log)
        m = ranking.metrics(texts, images)
        m |= {"description": desc, "package_mb": sizes["package_mb"], "download_mb": size_mb(zipped),
              "by_function": {"text": texts["by_function"], "image": images["by_function"]}}
        m["passes"] = passes(m, gates)
        m["placement_min_fraction"] = min(v["fraction"] for v in ane_placement(compiled, UNITS).values())
        feeds = {"embed_text_16": text_feed(np.array([2, 8071, 236787, 1], np.int32), pad_id)[1],
                 "embed_image_70_square": {"pixels": np.random.default_rng(0).random((1, 3, 384, 384), np.float32)}}
        m["latency_cpu_and_ne"] = {fn: latency_ms(p, fn, f) for fn, f in feeds.items()}
        variants[recipe] = m
        log(f"{recipe}: download {m['download_mb']:.0f} MB, min cosine {m['min_cosine']:.5f}, recall@10 "
            f"{m['min_recall_at_10']:.3f} (vs bucket {m['vs_bucket']['min_recall_at_10']:.3f}), p5 spearman "
            f"{m['min_p5_spearman_top50']:.3f} (vs bucket {m['vs_bucket']['min_p5_spearman_top50']:.3f}), "
            f"passes {m['passes']}")
        del p
        ranking.remove(compiled)
        if all(m["passes"].values()):
            winner = recipe
            break
        ranking.remove(pkg, zipped)
        gc.collect()

    report.details["variants"] = variants
    if winner is None:  # report the variant closest to passing (most gates met, then smallest)
        chosen = min(variants, key=lambda r: (-sum(variants[r]["passes"].values()), variants[r]["download_mb"]))
    else:
        chosen = winner
        final, final_zip = out / "EmbeddingGemma2.mlpackage", out / "EmbeddingGemma2.mlpackage.zip"
        ranking.remove(final, final_zip)
        (out / f"{winner}.mlpackage").rename(final)
        ranking.zip_package(final, final_zip)
        ranking.remove(out / f"{winner}.zip")
        (out / "checksums.json").write_text(json.dumps(ranking.checksums(final, final_zip), indent=1) + "\n")
    m = variants[chosen]
    report.metrics |= {k: m[k] for k in ("min_cosine", "min_recall_at_10", "min_p5_spearman_top50")}
    report.metrics["max_download_mb"] = m["download_mb"]
    report.details |= {"chosen": chosen, "winner": winner, "seconds": round(time.monotonic() - t0, 1)}
    shutil.rmtree(fp16, ignore_errors=True)
    log(f"done: chosen {chosen} (winner {winner})")


def main() -> None:
    run_stage(STAGE, fill)


if __name__ == "__main__":
    main()
