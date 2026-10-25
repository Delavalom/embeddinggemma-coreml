# group: convert
"""s5_swift: the Swift package end to end on macOS (egcm-parity) against s0: token ids, pixels, vectors.

Gates (gates.toml [s5_swift]): token_ids_identical, max_pixel_abs_diff, min_cosine. See stages/s5_swift/README.md.

    just stage 5                                                    # the stage (needs a passing, current s4)
    uv run --exact --group convert python -m stages.s5_swift.run --measure <pkg> [--compute-units cpuOnly] [--limit N]

`--measure` runs the same comparison on any package with the s5 I/O (for example the contract model of
make_contract_model.py) and prints the metrics without writing a report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from egcm.buckets import BUDGETS, choose_bucket
from egcm.config import HF_SNAPSHOT
from egcm.fixtures import S0_DIR, load_images, load_texts
from egcm.paths import ARTIFACTS, FIXTURES, HF_REVISION, ROOT, stage_dir
from egcm.report import Report
from egcm.stage import run_stage
from stages.s5_swift.compare import summarize
from stages.s5_swift.export_fixtures import SYNTHETIC_SIZES, synthetic, torch_resize

STAGE = "s5_swift"
S4_PACKAGE = ARTIFACTS / "s4_compress" / "EmbeddingGemma2.mlpackage"
SWIFT = ROOT / "swift"


def tokenizer_dir() -> Path:
    """s0's pinned HF snapshot, else the Swift tests' verified download (same revision)."""
    for d in (HF_SNAPSHOT, ARTIFACTS / "tokenizer" / HF_REVISION):
        if (d / "tokenizer.json").is_file() and (d / "tokenizer_config.json").is_file():
            return d
    raise FileNotFoundError(f"no tokenizer.json in {HF_SNAPSHOT} or artifacts/tokenizer/{HF_REVISION}: "
                            "run `just stage 0` (or `cd swift && swift test` to download it)")


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


def run_parity(model: Path, out: Path, synthetic_dir: Path, compute_units: str, limit: int) -> dict:
    """Build and run egcm-parity (release), return its JSON."""
    cmd = ["swift", "run", "-c", "release", "egcm-parity", "--model", str(model), "--tokenizer", str(tokenizer_dir()),
           "--fixtures", str(FIXTURES), "--out", str(out), "--synthetic", str(synthetic_dir),
           "--compute-units", compute_units, "--limit", str(limit)]
    print("$ " + " ".join(cmd), file=sys.stderr, flush=True)
    subprocess.run(cmd, cwd=SWIFT, check=True)
    return json.loads(out.read_text())


class _S0Image(dict):
    """s0's bucket, vector and (read on first use) uint8 bucket pixels for one image."""

    def __init__(self, z, image_id: str, row: int):
        super().__init__(bucket=str(z["bucket"][row]), emb=z["emb_bucket"][row])
        self._z, self._id = z, image_id

    def __getitem__(self, key):
        if key == "u8" and not super().__contains__("u8"):
            return self._z[f"pixels_{self._id}"]
        return super().__getitem__(key)


def compare(parity: dict, s0: Path, synthetic_refs: dict, limit: int = 0) -> tuple[dict, dict]:
    """Load s0's references for the fixtures egcm-parity covered and summarize."""
    texts, images = load_texts(), load_images()
    if limit:
        texts, images = texts[:limit], images[:limit]
    with np.load(s0 / "texts.npz") as z:
        order = list(z["order"])
        ref_texts = {t["id"]: {"ids": z[f"ids_{t['id']}"].tolist(), "emb": z["emb"][order.index(t["id"])],
                               "kind": t["kind"]} for t in texts}
    npz = {b: np.load(s0 / f"images_{b}.npz") for b in BUDGETS}
    try:
        ref_images = {}
        for b, z in npz.items():
            order = list(z["order"])
            ref_images[b] = {r["id"]: _S0Image(z, r["id"], order.index(r["id"])) for r in images}
        formats = {r["id"]: "png" if r["file"].lower().endswith(".png") else "jpeg" for r in images}
        return summarize(parity, ref_texts, ref_images, formats, synthetic_refs)
    finally:
        for z in npz.values():
            z.close()


def measure(model: Path, out_dir: Path, s0: Path = S0_DIR, compute_units: str = "cpuAndNeuralEngine",
            limit: int = 0) -> tuple[dict, dict]:
    if not (model / "Manifest.json").is_file():
        raise FileNotFoundError(f"{model} is not an .mlpackage: run `just stage 4`")
    t0 = time.monotonic()
    refs = write_synthetic(out_dir / "synthetic")
    parity = run_parity(model, out_dir / "parity.json", out_dir / "synthetic", compute_units, limit)
    metrics, details = compare(parity, s0, refs, limit)
    details |= {"model": str(model), "parity_json": str(out_dir / "parity.json"), "limit": limit,
                "seconds": round(time.monotonic() - t0, 1)}
    return metrics, details


def fill(report: Report) -> None:
    metrics, details = measure(S4_PACKAGE, stage_dir(STAGE))
    report.metrics |= metrics
    report.details |= details
    print(json.dumps(metrics), flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure", type=Path, help="compare this package without writing a report")
    ap.add_argument("--s0", type=Path, default=S0_DIR, help="s0_reference artifacts (with --measure)")
    ap.add_argument("--compute-units", default="cpuAndNeuralEngine", choices=["cpuAndNeuralEngine", "cpuOnly", "all"])
    ap.add_argument("--limit", type=int, default=0, help="first N texts and images only (with --measure)")
    ap.add_argument("--out", type=Path, default=ARTIFACTS / STAGE / "measure")
    a = ap.parse_args(argv)
    if a.measure is None:
        if (a.s0, a.compute_units, a.limit) != (S0_DIR, "cpuAndNeuralEngine", 0):
            sys.exit("--s0, --compute-units and --limit only apply with --measure; the stage always runs everything")
        run_stage(STAGE, fill)
        return
    metrics, details = measure(a.measure.resolve(), a.out, a.s0, a.compute_units, a.limit)
    print(json.dumps({"metrics": metrics, "details": details}, indent=1, default=float))


if __name__ == "__main__":
    main()
