"""s4's split (ADR-0010): the gated 14-function package -> one package per image budget (the four text functions and
that budget's five image functions) with one device archive each, and the check that the split changed nothing.

Both heavy steps run in a process of their own (16 GB Mac) under egcm.parts' memory guard:

    python -m stages.s4_compress.split extract <src> <out> <budget>
    python -m stages.s4_compress.split identity <src> <work> <budget>=<pkg> [<budget>=<pkg> ...]

- `extract`: the budget's functions of <src>, copied with egcm.mfmerge.merge. That is what coremltools'
  `MultiFunctionDescriptor` + `save_multifunction` do (scripts/probes/extract_functions.py, the probe ADR-0009
  measured), but streaming: one weight in memory at a time instead of every function's constants, each function's
  MIL copied unchanged, each distinct weight stored once.
- `identity`: every function of each budget package on one deterministic input (`feed`), `CPU_ONLY`, against the
  same function of <src>. The output bytes must be equal. One model loaded at a time, the E5 bundles it created
  deleted after each.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from egcm.buckets import BUCKETS, BUDGETS
from egcm.manifest import TEXT_FUNCTIONS, required_functions

STEM = "EmbeddingGemma2"


def names(budget: int) -> tuple[str, str]:
    """(package, device archive) file names of one budget in artifacts/s4_compress/."""
    return f"{STEM}-b{budget}.mlpackage", f"{STEM}-b{budget}.aar"


def feed(fn: str) -> dict[str, np.ndarray]:
    """One deterministic input for `fn` (the contract's types and shapes; egcm.functions)."""
    rng = np.random.default_rng(0)
    if fn in TEXT_FUNCTIONS:
        n = int(fn.rsplit("_", 1)[1])
        k = 3 * n // 4  # 3/4 tokens, the rest padding; ids below 4,000 also fit the tests' tiny model
        ids = rng.integers(3, 4_000, (1, n), dtype=np.int32)
        ids[0, 0], ids[0, k - 1], ids[0, k:] = 2, 1, 0
        return {"input_ids": ids, "attention_mask": (np.arange(n) < k).astype(np.int32)[None]}
    b = next(b for b in BUCKETS if b.function_name == fn)
    return {"pixels": rng.random((1, 3, b.height, b.width), dtype=np.float32)}


def function_names(pkg: Path) -> list[str]:
    from egcm import mfmerge

    return sorted(f.name for f in mfmerge.load_spec(pkg).description.functions)


def compare(ref: np.ndarray, got: np.ndarray) -> dict:
    """Bitwise equality of two outputs (NaN-safe: bytes, not values), and the largest difference for the record."""
    ref, got = np.asarray(ref, np.float32), np.asarray(got, np.float32)
    same = ref.shape == got.shape and ref.tobytes() == got.tobytes()
    diff = float(np.max(np.abs(ref - got))) if ref.shape == got.shape else float("inf")
    return {"identical": same, "max_abs_diff": diff}


def summarize(results: dict[str, dict[str, dict]]) -> dict:
    """budget -> function -> `compare` result  ->  the report's `details.split_identity`."""
    per = {b: {"identical": bool(fns) and all(r["identical"] for r in fns.values()), "functions": fns}
           for b, fns in results.items()}
    return {"identical": bool(per) and all(v["identical"] for v in per.values()), "budgets": per}


# --- children ---------------------------------------------------------------------------------------------------
def _extract(src: Path, out: Path, budget: int) -> dict:
    from egcm import mfmerge

    fns = required_functions(budget)
    return mfmerge.merge([(src, f, f) for f in fns], out, TEXT_FUNCTIONS[0])


def _identity(src: Path, work: Path, packages: dict[int, Path], predict: Callable | None = None) -> dict:
    """Compiles <src> and each package into `work`, runs every function of each package and the same function of
    <src> on `feed(fn)` (CPU_ONLY), and returns `summarize` of the comparisons."""
    from egcm.convert import compile_package

    predict = predict or _predict
    compiled = {"src": work / "src.mlmodelc"} | {b: work / f"b{b}.mlmodelc" for b in packages}
    try:
        compile_package(src, compiled["src"])
        fns = {}
        for b, p in packages.items():
            compile_package(p, compiled[b])
            fns[b] = function_names(p)
        results: dict[str, dict[str, dict]] = {str(b): {} for b in packages}
        for fn in sorted(set().union(*fns.values())):
            x = feed(fn)
            ref = predict(compiled["src"], fn, x)
            for b in packages:
                if fn in fns[b]:
                    results[str(b)][fn] = compare(ref, predict(compiled[b], fn, x))
                    print(f"[s4 split] b{b} {fn}: {results[str(b)][fn]}", file=sys.stderr, flush=True)
    finally:
        for c in compiled.values():
            shutil.rmtree(c, ignore_errors=True)
    return summarize(results)


def _predict(compiled: Path, fn: str, x: dict[str, np.ndarray]) -> np.ndarray:
    """One prediction with only this function loaded; the model and the E5 bundles it created are released after."""
    import coremltools as ct

    from egcm.e5cache import bundles

    before = bundles()
    m = ct.models.CompiledMLModel(str(compiled), compute_units=ct.ComputeUnit.CPU_ONLY, function_name=fn)
    y = np.array(m.predict(x)["embedding"], dtype=np.float32).reshape(-1)
    del m
    gc.collect()
    for p in bundles() - before:
        shutil.rmtree(p, ignore_errors=True)
    return y


# --- parent -----------------------------------------------------------------------------------------------------
def _child(args: list[str], tmp: Path) -> dict:
    """`python -m stages.s4_compress.split <args>` with its own TMPDIR; its JSON (last stdout line)."""
    from egcm.parts import GUARD_EXIT
    from egcm.paths import ROOT

    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    env = os.environ | {"TMPDIR": str(tmp) + "/", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)}
    try:
        p = subprocess.run([sys.executable, "-m", "stages.s4_compress.split", *args], env=env, cwd=ROOT,
                           stdout=subprocess.PIPE, text=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if p.returncode != 0:
        why = "memory guard" if p.returncode == GUARD_EXIT else f"exit {p.returncode}"
        raise RuntimeError(f"stages.s4_compress.split {args[0]} failed ({why}); see the log above")
    return json.loads(p.stdout.strip().splitlines()[-1])


def split(src: Path, out: Path, tokenizer: Path, log: Callable[[str], None] = print) -> dict[str, dict]:
    """Writes out/EmbeddingGemma2-b<budget>.mlpackage and .aar for each budget; returns `details.budgets`."""
    from egcm.archive import device_archive
    from egcm.convert import contract_problems, size_mb

    res = {}
    for b in BUDGETS:
        pkg_name, aar_name = names(b)
        pkg, want = out / pkg_name, sorted(required_functions(b))
        r = _child(["extract", str(src), str(pkg), str(b)], out / f".split_tmp_b{b}")
        if (got := function_names(pkg)) != want:
            raise AssertionError(f"{pkg_name} holds {got}, expected {want}")
        if problems := contract_problems(pkg, want):
            raise AssertionError(f"{pkg_name}: I/O contract mismatch: {problems}")
        aar = device_archive(pkg, tokenizer, out / aar_name)
        res[str(b)] = {"package": pkg_name, "device_archive": aar_name, "functions": want,
                       "package_mb": size_mb(pkg), "download_mb": size_mb(aar), "blobs_written": r["blobs_written"],
                       "peak_footprint_gb": r["peak_footprint_gb"]}
        log(f"budget {b}: {pkg_name} {res[str(b)]['package_mb']:.1f} MB, {aar_name} "
            f"{res[str(b)]['download_mb']:.1f} MB, {len(want)} functions")
    return res


def identity(src: Path, packages: dict[int, Path], work: Path, log: Callable[[str], None] = print) -> dict:
    """`details.split_identity`: every function of each budget package vs <src>, CPU_ONLY, bitwise."""
    args = ["identity", str(src), str(work), *(f"{b}={p}" for b, p in sorted(packages.items()))]
    res = _child(args, work.with_name(work.name + "_tmp"))
    log(f"split identical to the 14-function package (CPU_ONLY, bitwise): {res['identical']}")
    return res


def main(argv: list[str] | None = None) -> None:
    from egcm.parts import Guard, footprint_gb

    a = sys.argv[1:] if argv is None else argv
    guard = Guard()
    if a[0] == "extract":
        res = _extract(Path(a[1]), Path(a[2]), int(a[3]))
    elif a[0] == "identity":
        work = Path(a[2])
        work.mkdir(parents=True, exist_ok=True)
        packages = {int(b): Path(p) for b, p in (x.split("=", 1) for x in a[3:])}
        res = _identity(Path(a[1]), work, packages)
        shutil.rmtree(work, ignore_errors=True)
    else:
        raise SystemExit(f"unknown step {a[0]!r}")
    print(json.dumps(res | {"peak_footprint_gb": round(max(guard.peak, footprint_gb()), 2)}), flush=True)


if __name__ == "__main__":
    main()
