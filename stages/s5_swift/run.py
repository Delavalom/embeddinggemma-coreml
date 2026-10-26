# group: convert
"""s5_swift: the Swift package end to end on macOS (egcm-parity), against Python Core ML and the FP reference on the
same inputs (ADR-0008), plus token ids and pixels against s0.

Gates (gates.toml [s5_swift]): token_ids_identical, max_pixel_abs_diff, min_cosine_runtime, and s4's task-level
gates against the FP reference on Swift's pixels. See stages/s5_swift/README.md.

    just stage 5                                                    # the stage (needs passing, current s0, s1, s4)
    uv run --exact --group convert python -m stages.s5_swift.run --measure <pkg> [--compute-units cpuOnly] [--limit N]
        [--skip-fp]

The stage runs s4's two budget packages (ADR-0010; `--model <budget>=<pkg>` per budget to egcm-parity and the
runtime, stages/s5_swift/models.py). `--measure` runs the same comparison on any one package with the s5 I/O and all
14 functions (for example the contract model of make_contract_model.py) and prints the metrics without writing a
report. Three heavy steps, one process each, one
at a time: egcm-parity (Swift), stages.s5_swift.runtime (Python Core ML), stages.s5_swift.fp_reference per budget.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from egcm import manifest, published
from egcm.buckets import BUDGETS, choose_bucket
from egcm.fixtures import S0_DIR, load_images, load_texts
from egcm.paths import ARTIFACTS, FIXTURES, ROOT, stage_dir
from egcm.report import Report
from egcm.stage import run_stage
from stages.s5_swift import task_level as tl
from stages.s5_swift.compare import sha256_hwc, summarize
from stages.s5_swift.export_fixtures import SYNTHETIC_SIZES, synthetic, torch_resize
from stages.s5_swift.models import TEXT_BUDGET, model_args

STAGE = "s5_swift"
SWIFT = ROOT / "swift"
GB = 1e9
Model = Path | Mapping[int, Path]  # one package for every budget, or budget -> package (stages/s5_swift/models.py)


def write_synthetic(folder: Path) -> dict[str, dict]:
    """The lossless synthetic pattern at SYNTHETIC_SIZES as PNGs for egcm-parity, and torch's resize of each as the
    reference: "syn_<h>x<w>_<budget>" -> {"bucket", "u8"}."""
    folder.mkdir(parents=True, exist_ok=True)
    refs = {}
    for h, w in SYNTHETIC_SIZES:
        name, u8 = f"syn_{h}x{w}", synthetic(h, w)
        Image.fromarray(u8, "RGB").save(folder / f"{name}.png")
        for b in BUDGETS:
            bk = choose_bucket(h, w, b)
            refs[f"{name}_{b}"] = {"bucket": bk.name, "u8": torch_resize(u8, bk.height, bk.width)}
    return refs


def run_parity(model: Model, out: Path, synthetic_dir: Path, compute_units: str, limit: int,
               tokenizer: Path | None = None, pixels: Path | None = None) -> dict:
    """Build and run egcm-parity (release), return its JSON. `model`: one package for every budget, or budget ->
    package (`--model <budget>=<pkg>` per budget, ADR-0010). `tokenizer` defaults to the verified local copy."""
    cmd = ["swift", "run", "-c", "release", "egcm-parity", *model_args(model),
           "--tokenizer", str(tokenizer or published.tokenizer_source()), "--fixtures", str(FIXTURES),
           "--out", str(out), "--synthetic", str(synthetic_dir), "--compute-units", compute_units,
           "--limit", str(limit)]
    if pixels is not None:
        cmd += ["--pixels", str(pixels)]
    print("$ " + " ".join(cmd), file=sys.stderr, flush=True)
    subprocess.run(cmd, cwd=SWIFT, check=True)
    return json.loads(out.read_text())


def run_part(module: str, args: list[str], out: Path) -> dict[str, np.ndarray]:
    """`python -m stages.s5_swift.<module> ... --out <out>` in its own process; its npz as a dict."""
    cmd = [sys.executable, "-m", f"stages.s5_swift.{module}", *args, "--out", str(out)]
    print("$ " + " ".join(cmd), file=sys.stderr, flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    with np.load(out) as z:
        return {k: z[k] for k in z.files}


class _S0Image(dict):
    """s0's bucket, vector and (read on first use) uint8 bucket pixels for one image."""

    def __init__(self, z, image_id: str, row: int):
        super().__init__(bucket=str(z["bucket"][row]), emb=z["emb_bucket"][row])
        self._z, self._id = z, image_id

    def __getitem__(self, key):
        if key == "u8" and not super().__contains__("u8"):
            return self._z[f"pixels_{self._id}"]
        return super().__getitem__(key)


def fp_needed(parity: dict, ref_images: dict) -> dict[int, list[str]]:
    """Per budget, the image ids whose Swift bucket image is not byte-identical to s0's (they need the FP32 rewrite)."""
    return {b: sorted(i for i, r in refs.items() if f"{i}_{b}" in parity["images"]
                      and parity["images"][f"{i}_{b}"]["sha256"] != sha256_hwc(r["u8"]))
            for b, refs in ref_images.items()}


def compare(parity: dict, s0: Path, synthetic_refs: dict, limit: int = 0, py: dict | None = None,
            fp: dict[int, dict] | None = None) -> tuple[dict, dict]:
    """Every s5 metric from what the three steps wrote. `py`: the runtime npz; `fp`: budget -> fp_reference npz
    (None skips the task level, `--skip-fp`)."""
    texts, images = load_texts(), load_images()
    if limit:
        texts, images = texts[:limit], images[:limit]
    with np.load(s0 / "texts.npz") as z:
        order = list(z["order"])
        ref_texts = {t["id"]: {"ids": z[f"ids_{t['id']}"].tolist(), "emb": z["emb"][order.index(t["id"])],
                               "kind": t["kind"]} for t in texts}
        s0_texts = {"order": [str(k) for k in order], "emb": z["emb"]}
    npz = {b: np.load(s0 / f"images_{b}.npz") for b in BUDGETS}
    try:
        ref_images, s0_images = {}, {}
        for b, z in npz.items():
            order = list(z["order"])
            ref_images[b] = {r["id"]: _S0Image(z, r["id"], order.index(r["id"])) for r in images}
            s0_images[b] = {"order": [str(k) for k in order], "emb_bucket": z["emb_bucket"], "emb_raw": z["emb_raw"]}
        kinds, formats = tl.kinds_and_formats(texts, images)
        metrics, details = summarize(parity, ref_texts, ref_images, formats, synthetic_refs)
        need = fp_needed(parity, ref_images)
    finally:
        for z in npz.values():
            z.close()
    metrics["min_cosine_runtime"], details["runtime"] = tl.runtime(parity, py or {}, kinds, formats)
    if fp is None:
        return metrics, details
    for b in BUDGETS:
        got = sorted(k for k in fp[b] if k != "peak_rss_bytes")
        if got != need[b]:
            raise RuntimeError(f"fp_reference {b}: computed {len(got)} images, {len(need[b])} differ from s0")
    sw_texts, sw_images = tl.swift_vectors(parity, s0_texts, s0_images)
    refs = {b: tl.image_reference(v["order"], v["ref_bucket"], fp[b]) for b, v in sw_images.items()}
    gated, task = tl.task_level(sw_texts, sw_images, refs, need)
    return metrics | gated, details | task


def measure(model: Model, out_dir: Path, s0: Path = S0_DIR, compute_units: str = "cpuAndNeuralEngine",
            limit: int = 0, tokenizer: Path | None = None, skip_fp: bool = False) -> tuple[dict, dict]:
    """`model`: one package for every budget (`--measure`), or budget -> package (the stage, ADR-0010)."""
    for p in model.values() if isinstance(model, Mapping) else [model]:
        if not (p / "Manifest.json").is_file():
            raise FileNotFoundError(f"{p} is not an .mlpackage: run `just stage 4`")
    if limit and limit < tl.MIN_IMAGES and not skip_fp:
        raise ValueError(f"--limit {limit} is below {tl.MIN_IMAGES} images, which the top-50 ranking metrics need: "
                         "add --skip-fp")
    t0, rss = time.monotonic(), {}
    refs = write_synthetic(out_dir / "synthetic")
    pixels, parity_json = out_dir / "pixels", out_dir / "parity.json"
    parity = run_parity(model, parity_json, out_dir / "synthetic", compute_units, limit, tokenizer, pixels)
    rss["swift_gb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / GB  # bytes on macOS
    common = ["--parity", str(parity_json), "--pixels", str(pixels)]
    py = run_part("runtime", [*model_args(model), "--compute-units", compute_units, *common], out_dir / "runtime.npz")
    rss["runtime_gb"] = float(py.pop("peak_rss_bytes")) / GB
    fp = None
    if not skip_fp:
        fp = {}
        for b in BUDGETS:
            fp[b] = run_part("fp_reference", ["--budget", str(b), "--s0", str(s0), *common], out_dir / f"fp_{b}.npz")
            rss[f"fp_{b}_gb"] = float(fp[b]["peak_rss_bytes"]) / GB
    metrics, details = compare(parity, s0, refs, limit, py, fp)
    models = {str(b): str(p) for b, p in model.items()} if isinstance(model, Mapping) else str(model)
    # egcm-parity's own record of what it loaded: `model_revisions` per budget ("70", "280"); it embeds text with
    # budget 70's model, as runtime.py does (models.TEXT_BUDGET)
    details |= {"model": models, "text_budget": TEXT_BUDGET if isinstance(model, Mapping) else None,
                "swift_model_revisions": parity.get("model_revisions"),
                "parity_json": str(parity_json), "limit": limit, "peak_rss": rss,
                "seconds": round(time.monotonic() - t0, 1)}
    return metrics, details


def fill(report: Report) -> None:
    """Runs on s4's device archives, one per image budget, unpacked (sha256-checked) into unpacked/b<budget>/: the
    package and tokenizer bytes a phone gets (ADR-0010). The tokenizer is the same in both; s5 uses TEXT_BUDGET's."""
    s4, out = published.shipped(), stage_dir(STAGE)
    models, tokenizers = {}, {}
    for b, v in sorted(s4.budgets.items()):
        unpacked = manifest.unpack(v.archive, out / "unpacked" / published.budget_dir(b), v.archive_sha256)
        tokenizers[b] = published.check_tokenizer(unpacked / published.TOKENIZER_DIR)
        models[b] = unpacked / published.PACKAGE_NAME
    metrics, details = measure(models, out, tokenizer=tokenizers[TEXT_BUDGET])
    report.metrics |= metrics
    report.details |= details | {"s4_recipe": s4.recipe, "device_archives": {
        str(b): {"name": v.archive.name, "sha256": v.archive_sha256} for b, v in sorted(s4.budgets.items())}}
    print(json.dumps(metrics), flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure", type=Path, help="compare this package without writing a report")
    ap.add_argument("--s0", type=Path, default=S0_DIR, help="s0_reference artifacts (with --measure)")
    ap.add_argument("--compute-units", default="cpuAndNeuralEngine", choices=["cpuAndNeuralEngine", "cpuOnly", "all"])
    ap.add_argument("--limit", type=int, default=0, help="first N texts and images only (with --measure)")
    ap.add_argument("--skip-fp", action="store_true", help="no FP reference, no task-level metrics (with --measure)")
    ap.add_argument("--out", type=Path, default=ARTIFACTS / STAGE / "measure")
    a = ap.parse_args(argv)
    if a.measure is None:
        if (a.s0, a.compute_units, a.limit, a.skip_fp) != (S0_DIR, "cpuAndNeuralEngine", 0, False):
            sys.exit("--s0, --compute-units, --limit and --skip-fp only apply with --measure; the stage always runs "
                     "everything")
        run_stage(STAGE, fill)
        return
    metrics, details = measure(a.measure.resolve(), a.out, a.s0, a.compute_units, a.limit, skip_fp=a.skip_fp)
    print(json.dumps({"metrics": metrics, "details": details}, indent=1, default=float))


if __name__ == "__main__":
    main()
