# group: convert
"""s4_compress: compressed variants of the FP16 model; keeps the smallest that passes every s4 gate.

Gates (gates.toml [s4_compress], ADR-0007): min_cosine, min_recall_at_10 and min_p5_spearman_top50 (against the
bucketed FP reference), min_t2i_top1_in_top10 and max_i2i_label_p10_drop (bucketing, against exact-aspect),
max_download_mb. See stages/s4_compress/README.md.
"""

from __future__ import annotations

import json
import time

import coremltools as ct
import numpy as np

from egcm import parts
from egcm.archive import device_archive
from egcm.compress import MEASURE_ONLY, RECIPES
from egcm.config import HF_SNAPSHOT, load
from egcm.convert import compile_package, contract_problems, disk_free_gb, size_mb
from egcm.e5cache import PurgingPackage
from egcm.e5prune import BundlePruner
from egcm.evaluate import latency_ms, run_images, run_texts, text_feed
from egcm.functions import FUNCTIONS
from egcm.gates import gates as stage_gates
from egcm.gates import op_for, targets
from egcm.paths import stage_dir
from egcm.placement import ane_placement_per_function
from egcm.ranking import metrics as task_metrics
from egcm.report import Report
from egcm.stage import run_stage
from stages.s4_compress import ranking

STAGE = "s4_compress"
UNITS = ct.ComputeUnit.CPU_AND_NE  # the shipping configuration


# gate key -> variant metric key, where they differ
METRIC_FOR_GATE = {"max_download_mb": "download_mb"}
# recorded-only target -> vs_raw (exact-aspect) metric (ADR-0007)
TARGET_METRIC = {"t2i_recall_at_10_exact_aspect": "min_recall_at_10"}


def passes(m: dict, gates: dict) -> dict[str, bool]:
    """Every gate of gates.toml [s4_compress] against the variant's metrics, in the key's direction."""
    out = {}
    for key, gate in gates.items():
        value, op = m[METRIC_FOR_GATE.get(key, key)], op_for(key, gate)
        out[key] = value == gate if op == "==" else value >= gate if op == ">=" else value <= gate
    return out


def evaluate_variant(recipe: str, desc: str, sizes: dict, out, pad_id: int, gates: dict, log,
                     placement: bool = True) -> dict:
    """Metrics for out/<recipe>.mlpackage on CPU_AND_NE. `placement=False` skips the Neural Engine placement
    analysis (about 90 minutes for 14 functions; scripts.measure_s4_eval)."""
    pkg, compiled, zipped = out / f"{recipe}.mlpackage", out / f"{recipe}.mlmodelc", out / f"{recipe}.zip"
    if problems := contract_problems(pkg, FUNCTIONS):
        raise AssertionError(f"{recipe}: I/O contract mismatch: {problems}")
    aar = device_archive(pkg, HF_SNAPSHOT, out / f"{recipe}.aar")  # the device download (package + tokenizer)
    ranking.zip_package(pkg, zipped)  # the package alone, deflate: recorded for comparison (zip_mb), not gated
    compile_package(pkg, compiled)
    p = PurgingPackage(compiled, UNITS)
    texts, images = run_texts(p, pad_id), run_images(p, log=log)
    m = task_metrics(texts, images)
    m |= {"description": desc, "package_mb": sizes["package_mb"], "download_mb": size_mb(aar),
          "zip_mb": size_mb(zipped),
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
    ranking.remove(zipped)
    log(f"{recipe}: download (Apple Archive, LZFSE, with tokenizer) {m['download_mb']:.1f} MB (package zip "
        f"{m['zip_mb']:.1f} MB), min cosine {m['min_cosine']:.5f}, recall@10 "
        f"{m['min_recall_at_10']:.3f}, p5 spearman {m['min_p5_spearman_top50']:.3f} (vs bucketed FP); top-1 in top 10 "
        f"{m['min_t2i_top1_in_top10']:.3f}, i2i p10 drop {m['max_i2i_label_p10_drop']:.3f}, recall@10 vs exact-aspect "
        f"{m['vs_raw']['min_recall_at_10']:.3f}; passes {m['passes']}")
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
            ranking.remove(out / f"{recipe}.mlpackage", out / f"{recipe}.aar")
    final, final_aar = out / "EmbeddingGemma2.mlpackage", out / "EmbeddingGemma2.aar"
    ranking.remove(final, final_aar, out / "EmbeddingGemma2.mlpackage.zip", out / "checksums.json")
    (out / f"{chosen}.mlpackage").rename(final)
    if winner:  # the shippable package and its device download (the archive names the package EmbeddingGemma2)
        (out / f"{chosen}.aar").rename(final_aar)
        (out / "checksums.json").write_text(json.dumps(ranking.checksums(final, final_aar), indent=1) + "\n")
    else:
        ranking.remove(out / f"{chosen}.aar")
    m = variants[chosen]
    report.metrics |= {k: m[METRIC_FOR_GATE.get(k, k)] for k in gates}
    # recorded only (gates.toml [s4_compress.targets]): text->image recall@10 against exact-aspect (ADR-0007)
    report.details["targets"] = {k: {"target": v, "value": m["vs_raw"][TARGET_METRIC[k]]}
                                 for k, v in targets(STAGE).items()}
    report.details |= {"chosen": chosen, "winner": winner, "package": final.name,
                       "device_archive": final_aar.name if winner else None, "zip_mb": m["zip_mb"],
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
            ranking.remove(out / f"{r}.mlpackage", out / f"{r}.aar")
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
