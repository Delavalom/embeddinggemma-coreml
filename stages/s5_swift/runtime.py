"""Python Core ML on Swift's exact inputs (ADR-0008): the runtime-parity side of s5. One heavy process.

    python -m stages.s5_swift.runtime --model 70=<pkg> --model 280=<pkg> --parity <parity.json> --pixels <dir>
        --out <npz> [--compute-units cpuAndNeuralEngine|cpuOnly|all]
    python -m stages.s5_swift.runtime --model <pkg> ...          # one 14-function package for every budget

The same packages Swift ran, compiled by coremltools, on the same compute units, one package compiled and one
function loaded at a time (as s4 evaluates). Image vectors of each budget come from that budget's package, text
vectors from `models.TEXT_BUDGET`'s (stages/s5_swift/models.py). Inputs are what the Swift package fed its model: its
token ids (parity.json), padded to the smallest text function that fits as `egcm.evaluate.text_feed` does, and its
bucket images (`--pixels`, sha256-checked). Writes `text_<id>` and `image_<id>_<budget>` (raw model outputs) and
`peak_rss_bytes`.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
from pathlib import Path

import coremltools as ct
import numpy as np

from egcm.config import HF_SNAPSHOT, load
from egcm.convert import compile_package
from egcm.e5cache import PurgingPackage
from egcm.e5prune import BundlePruner
from egcm.evaluate import text_feed
from stages.s5_swift.models import budget_of, parse_models
from stages.s5_swift.pixels import bucket_for, model_input, swift_pixels

UNITS = {"cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE, "cpuOnly": ct.ComputeUnit.CPU_ONLY,
         "all": ct.ComputeUnit.ALL}


def jobs(parity: dict, pad_id: int) -> list[tuple[str, str, object]]:
    """(function, output key, feed or image key) for every text and image, sorted by function (one load each)."""
    out = []
    for k, t in parity["texts"].items():
        fn, feed = text_feed(np.asarray(t["ids"], np.int32), pad_id)
        out.append((fn, f"text_{k}", feed))
    for k, e in parity["images"].items():
        out.append((bucket_for(e["bucket"], int(e["budget"])).function_name, f"image_{k}", k))
    return sorted(out, key=lambda j: (j[0], j[1]))


def by_package(models: dict[int, Path], todo: list[tuple[str, str, object]]) -> dict[Path, list]:
    """The jobs grouped by the package that runs them (`models.budget_of`), so each package is compiled once."""
    out: dict[Path, list] = {}
    for job in todo:
        out.setdefault(models[budget_of(job[0])], []).append(job)
    return out


def run(models: dict[int, Path], parity: dict, pixels: Path, compute_units: str, work: Path,
        log=print) -> dict[str, np.ndarray]:
    pad_id = load(HF_SNAPSHOT)[0].pad_id
    out: dict[str, np.ndarray] = {}
    with BundlePruner():  # Core ML's E5 cache grows while a function stays loaded (egcm.e5prune)
        for model, todo in by_package(models, jobs(parity, pad_id)).items():
            log(f"[s5 runtime] {model}")
            compiled = compile_package(model, work / "runtime.mlmodelc")
            try:
                pkg = PurgingPackage(compiled, UNITS[compute_units])
                current = None
                for fn, key, what in todo:
                    if fn != current:
                        log(f"[s5 runtime] {fn}")
                        current = fn
                    if isinstance(what, str):  # an image key
                        what = {"pixels": model_input(swift_pixels(parity["images"][what], pixels, what))}
                    out[key] = pkg.embed(fn, what)
                pkg.close()
            finally:
                shutil.rmtree(compiled, ignore_errors=True)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m stages.s5_swift.runtime")
    ap.add_argument("--model", action="append", required=True, help="<budget>=<pkg> per budget, or one <pkg>")
    ap.add_argument("--parity", type=Path, required=True)
    ap.add_argument("--pixels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compute-units", default="cpuAndNeuralEngine", choices=sorted(UNITS))
    a = ap.parse_args(argv)
    out = run(parse_models(a.model), json.loads(a.parity.read_text()), a.pixels, a.compute_units, a.out.parent,
              log=lambda s: print(s, file=sys.stderr, flush=True))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    np.savez(a.out, peak_rss_bytes=np.int64(rss), **out)


if __name__ == "__main__":
    main()
