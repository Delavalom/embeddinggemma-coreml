"""Process-isolated, resumable conversion of the entry points into multifunction packages (s2, s3, s4).

This Mac has 16 GB of RAM shared with other apps, so a stage never holds more than one part:
- **One part per process** (`python -m egcm.parts convert ...`). The child loads only the weights its function needs,
  traces, frees the torch module, converts, optionally compresses with each recipe, saves, and exits; its memory and
  coremltools' temporary packages (its own TMPDIR, deleted by the parent) go with it.
- **One merge per process** (`python -m egcm.parts merge ...`): the package so far + one part -> the new package,
  with the streaming merge in egcm.mfmerge (constant memory; weights with identical values are stored once). No
  torch in that process.
- **Resumable**: `build.json` next to the packages records the stage's code hash, each package's merged functions and
  its content sha256, and each saved part's sha256. A rerun with the same code skips merged functions and reuses saved
  parts whose sha matches; different code starts over.
- **Guards**: a child aborts (exit 70) if its physical footprint (resident + compressed) passes `FOOTPRINT_LIMIT_GB`,
  or system swap grows by more than `SWAP_GROWTH_LIMIT_GB` while it runs (the swap baseline on this Mac is already
  about 12 GB before any conversion, so an absolute limit would never let a part start). The parent refuses a step
  that would leave less than `MIN_FREE_GB` free on disk.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

FOOTPRINT_LIMIT_GB = 9.0
SWAP_GROWTH_LIMIT_GB = 6.0
MIN_FREE_GB = 25.0
GUARD_EXIT = 70


# --- measurement ----------------------------------------------------------------------------------------------
def footprint_gb(pid: int | None = None) -> float:
    """Physical footprint (resident + compressed) of a process, from proc_pid_rusage (RUSAGE_INFO_V0)."""
    buf = (ctypes.c_uint64 * 12)()
    if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(pid or os.getpid(), 0, ctypes.byref(buf)) != 0:
        return 0.0
    return buf[9] / 1e9


def swap_used_gb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) / 1e3 if m else 0.0


def disk_free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def tree_sha(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(path).rglob("*")):
        if p.is_file():
            h.update(p.relative_to(path).as_posix().encode())
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
    return h.hexdigest()


def size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e6


class Guard:
    """In a child: tracks peak footprint every 0.5 s and exits the process if a limit is crossed."""

    def __init__(self):
        self.peak = 0.0
        self.swap0 = swap_used_gb()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        n = 0
        while True:
            fp = footprint_gb()
            self.peak = max(self.peak, fp)
            swap = swap_used_gb() - self.swap0 if n % 4 == 0 else 0.0
            if fp > FOOTPRINT_LIMIT_GB or swap > SWAP_GROWTH_LIMIT_GB:
                print(f"memory guard: footprint {fp:.2f} GB (limit {FOOTPRINT_LIMIT_GB}), swap +{swap:.2f} GB "
                      f"(limit {SWAP_GROWTH_LIMIT_GB})", file=sys.stderr, flush=True)
                os._exit(GUARD_EXIT)
            n += 1
            time.sleep(0.5)


# --- child: convert one function ------------------------------------------------------------------------------
def _load(needs_vision: bool):
    """Only the weights this function needs, FP32: text tower + scaled table, and the vision encoder for images."""
    from safetensors import safe_open

    from egcm.config import HF_SNAPSHOT, load
    from egcm.weights import build_text_tower, build_vision_encoder, destination, text_table

    tcfg, vcfg, _ = load(HF_SNAPSHOT)
    towers = {"text", "table"} | ({"vision"} if needs_vision else set())
    sd = {}
    with safe_open(HF_SNAPSHOT / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            d = destination(k)
            if d is not None and d[0] in towers:
                sd[k.removeprefix("model.")] = f.get_tensor(k).float()
    tower, table = build_text_tower(tcfg, sd), text_table(sd, tcfg.hidden)
    encoder = build_vision_encoder(vcfg, sd, tcfg.hidden) if needs_vision else None
    del sd
    return tower, table, encoder, tcfg


def _convert_child(name: str, precision: str, out: Path, recipes: list[str]) -> dict:
    import coremltools as ct
    import numpy as np
    import torch

    from egcm.convert import TARGET, input_specs
    from egcm.fixtures import S0_DIR
    from egcm.functions import TEXT_FUNCTIONS, TextFunction, bucket_for, image_embedder, text_length
    from egcm.model.embedder import TextEmbedder

    guard = Guard()
    torch.set_grad_enabled(False)
    is_text = name in TEXT_FUNCTIONS
    tower, table, encoder, _ = _load(needs_vision=not is_text)
    if is_text:
        n = text_length(name)
        module = TextFunction(TextEmbedder(tower, table, n)).eval()
        example = (torch.zeros(1, n, dtype=torch.int32), torch.ones(1, n, dtype=torch.int32))
    else:
        b = bucket_for(name)
        layout = json.loads((S0_DIR / "image_token_layout.json").read_text())
        module = image_embedder(encoder, tower, table, layout, b)
        example = (torch.rand(1, 3, b.height, b.width, generator=torch.Generator().manual_seed(0)),)
    traced = torch.jit.trace(module, example)
    del module, tower, table, encoder, example  # the traced graph keeps only the tensors it uses
    gc.collect()
    m = ct.convert(traced, inputs=input_specs(name), outputs=[ct.TensorType("embedding", dtype=np.float32)],
                   convert_to="mlprogram", minimum_deployment_target=TARGET,
                   compute_precision={"fp32": ct.precision.FLOAT32, "fp16": ct.precision.FLOAT16}[precision],
                   compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
    del traced
    gc.collect()
    saved = {}
    if recipes:
        from egcm.compress import RECIPES

        for r in recipes:
            c = RECIPES[r][1](m)
            dest = out / r / f"{name}.mlpackage"
            dest.parent.mkdir(parents=True, exist_ok=True)
            c.save(str(dest))
            del c
            gc.collect()
            saved[r] = str(dest)
    else:
        dest = out / f"{name}.mlpackage"
        m.save(str(dest))
        saved[""] = str(dest)
    return {"parts": saved, "peak_footprint_gb": round(max(guard.peak, footprint_gb()), 2)}


# --- child: merge one part into a package ---------------------------------------------------------------------
def _merge_child(into: Path, part: Path, name: str) -> dict:
    """The package so far + one part -> the new package, with the streaming merge (egcm.mfmerge: constant memory;
    coremltools' save_multifunction needs memory for every function's weights)."""
    from egcm import mfmerge
    from egcm.functions import DEFAULT_FUNCTION

    guard = Guard()
    sources = []
    if into.exists():
        sources = [(into, f.name, f.name) for f in mfmerge.load_spec(into).description.functions]
    sources.append((part, "main", name))
    res = mfmerge.merge(sources, into, DEFAULT_FUNCTION)
    return res | {"peak_footprint_gb": round(max(guard.peak, footprint_gb()), 2)}


# --- parent -----------------------------------------------------------------------------------------------------
def _run(args: list[str], tmp: Path) -> dict:
    """Run `python -m egcm.parts <args>` with its own TMPDIR; returns the JSON on its last stdout line."""
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    from egcm.paths import ROOT

    env = os.environ | {"TMPDIR": str(tmp) + "/", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)}
    try:
        p = subprocess.run([sys.executable, "-m", "egcm.parts", *args], env=env, cwd=ROOT, capture_output=True,
                           text=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if p.returncode != 0:
        why = "memory guard" if p.returncode == GUARD_EXIT else f"exit {p.returncode}"
        raise RuntimeError(f"egcm.parts {args[0]} {args[2] if len(args) > 2 else ''} failed ({why}): "
                           f"{p.stderr[-2500:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def _need_disk(path: Path, gb: float, what: str) -> None:
    free = disk_free_gb(path)
    if free - gb < MIN_FREE_GB:
        raise RuntimeError(f"{what} needs about {gb:.1f} GB; {free:.1f} GB free would leave less than "
                           f"{MIN_FREE_GB:.0f} GB")


def build(stage: str, names: Iterable[str], precision: str, out: Path, recipes: list[str] | None = None,
          log: Callable[[str], None] = print) -> dict:
    """Build `out/<key>.mlpackage` for each key (each recipe, or "EmbeddingGemma2" without recipes), resumably.

    Returns {"packages": {key: {"path", "mb", "functions"}}, "part_mb": {"<key>/<name>": MB},
    "peak_footprint_gb": {"convert/<name>": GB, "merge/<key>/<name>": GB}}."""
    from egcm.provenance import code_hash

    names = list(names)
    keys = list(recipes) if recipes else ["EmbeddingGemma2"]
    parts_dir, state_path = out / "parts", out / "build.json"
    want_hash = code_hash(stage)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("code_hash") != want_hash or state.get("precision") != precision or state.get("keys") != keys:
        log("build: starting over (no state, or code / precision / recipes changed)")
        for k in keys:
            shutil.rmtree(out / f"{k}.mlpackage", ignore_errors=True)
        shutil.rmtree(parts_dir, ignore_errors=True)
        state = {"code_hash": want_hash, "precision": precision, "keys": keys, "packages": {}, "parts": {},
                 "part_mb": {}, "peak_footprint_gb": {}}
    for k in keys:  # a package whose content no longer matches its recorded sha is rebuilt from scratch
        pkg, rec = out / f"{k}.mlpackage", state["packages"].get(k)
        if rec and not (pkg.exists() and tree_sha(pkg) == rec["sha256"]):
            log(f"build: {pkg.name} does not match build.json; rebuilding it")
            rec = None
        if rec is None:
            shutil.rmtree(pkg, ignore_errors=True)
            state["packages"][k] = {"functions": [], "sha256": None}

    def save() -> None:
        state_path.write_text(json.dumps(state, indent=1) + "\n")

    def part_path(k: str, name: str) -> Path:
        return parts_dir / f"{name}.mlpackage" if not recipes else parts_dir / k / f"{name}.mlpackage"

    save()
    for name in names:
        todo = [k for k in keys if name not in state["packages"][k]["functions"]]
        if not todo:
            continue
        missing = [k for k in todo if not (part_path(k, name).exists()
                                           and state["parts"].get(f"{k}/{name}", {}).get("sha256")
                                           == tree_sha(part_path(k, name)))]
        if missing:
            last = max(state["part_mb"].values(), default=1200.0) / 1e3
            _need_disk(out, 2.5 * last * max(1, len(keys)), f"converting {name}")
            t = time.monotonic()
            res = _run(["convert", "--function", name, "--precision", precision, "--out", str(parts_dir),
                        "--recipes", ",".join(recipes or [])], out / "tmp")
            state["peak_footprint_gb"][f"convert/{name}"] = res["peak_footprint_gb"]
            for k in keys:
                p = part_path(k, name)
                state["parts"][f"{k}/{name}"] = {"mb": round(size_mb(p), 1), "sha256": tree_sha(p)}
                state["part_mb"][f"{k}/{name}"] = state["parts"][f"{k}/{name}"]["mb"]
            save()
            log(f"{name}: converted in {time.monotonic() - t:.0f} s, peak {res['peak_footprint_gb']} GB, parts "
                + ", ".join(f"{state['parts'][f'{k}/{name}']['mb']:.0f} MB" for k in keys))
        for k in todo:
            pkg, p = out / f"{k}.mlpackage", part_path(k, name)
            _need_disk(out, 2 * (size_mb(pkg) / 1e3 if pkg.exists() else 0) + size_mb(p) / 1e3, f"merging {name}")
            t = time.monotonic()
            res = _run(["merge", "--into", str(pkg), "--part", str(p), "--function", name], out / "tmp")
            state["peak_footprint_gb"][f"merge/{k}/{name}"] = res["peak_footprint_gb"]
            state["packages"][k] = {"functions": res["functions"], "sha256": tree_sha(pkg)}
            shutil.rmtree(p)
            state["parts"].pop(f"{k}/{name}", None)
            save()
            log(f"{pkg.name}: +{name} -> {len(res['functions'])} functions, {size_mb(pkg):.0f} MB "
                f"({time.monotonic() - t:.0f} s, peak {res['peak_footprint_gb']} GB)")
    shutil.rmtree(parts_dir, ignore_errors=True)
    return {"packages": {k: {"path": str(out / f"{k}.mlpackage"), "mb": round(size_mb(out / f"{k}.mlpackage"), 1),
                             "functions": state["packages"][k]["functions"]} for k in keys},
            "part_mb": state["part_mb"], "peak_footprint_gb": state["peak_footprint_gb"]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m egcm.parts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--function", required=True)
    c.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--recipes", default="")
    m = sub.add_parser("merge")
    m.add_argument("--into", type=Path, required=True)
    m.add_argument("--part", type=Path, required=True)
    m.add_argument("--function", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        res = _convert_child(a.function, a.precision, a.out, [r for r in a.recipes.split(",") if r])
    else:
        res = _merge_child(a.into, a.part, a.function)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
