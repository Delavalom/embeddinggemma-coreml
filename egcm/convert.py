"""torch -> Core ML for the entry points in egcm.functions, and the multifunction merge with shared weights.

Each function is traced and converted on its own (iOS18 / macOS15 ML program) and saved as a part. Parts are merged
with `MultiFunctionDescriptor` + `save_multifunction`, which deduplicates constants with identical values, so the
towers' weights (identical in every part) are stored once. Disk is bounded by merging every `batch` parts into the
growing package and deleting them. `save_multifunction` also writes a temporary copy of the weights, so the peak is
about package + `batch` parts + temporary weights + new package (FP32, batch 2: about 7.5 GB). Before every part and
every merge the free space is checked against that estimate (`DiskSpaceError` instead of a failed write).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from egcm.functions import DEFAULT_FUNCTION, EMBEDDING_DIM, TEXT_FUNCTIONS, bucket_for, text_length

TARGET = ct.target.iOS18
MARGIN_GB = 2.0  # kept free beyond the estimate (other processes share the disk)


class DiskSpaceError(RuntimeError):
    """Not enough free disk for the next conversion step."""


def input_specs(name: str) -> list[ct.TensorType]:
    """The contract inputs of one entry point (see egcm.functions)."""
    if name in TEXT_FUNCTIONS:
        n = text_length(name)
        return [ct.TensorType("input_ids", shape=(1, n), dtype=np.int32),
                ct.TensorType("attention_mask", shape=(1, n), dtype=np.int32)]
    b = bucket_for(name)
    return [ct.TensorType("pixels", shape=(1, 3, b.height, b.width), dtype=np.float32)]


def convert(name: str, module: torch.nn.Module, example: tuple[torch.Tensor, ...], precision) -> ct.models.MLModel:
    with torch.no_grad():
        traced = torch.jit.trace(module.eval(), example)
    return ct.convert(
        traced, inputs=input_specs(name), outputs=[ct.TensorType("embedding", dtype=np.float32)],
        convert_to="mlprogram", minimum_deployment_target=TARGET, compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,  # only the conversion-time load; callers choose units when loading
        skip_model_load=True,
    )


def _merge(sources: list[tuple[Path, str]], out: Path) -> None:
    """sources: (package, function name in it; "main" for a single-function part named after the file)."""
    desc = ct.utils.MultiFunctionDescriptor()
    names = []
    for pkg, fn in sources:
        target = pkg.stem if fn == "main" else fn
        desc.add_function(str(pkg), src_function_name=fn, target_function_name=target)
        names.append(target)
    desc.default_function_name = DEFAULT_FUNCTION if DEFAULT_FUNCTION in names else names[0]
    tmp = out.with_name(out.stem + ".merging.mlpackage")
    shutil.rmtree(tmp, ignore_errors=True)
    ct.utils.save_multifunction(desc, str(tmp))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)


def _need(path: Path, gb: float, what: str) -> None:
    free = disk_free_gb(path)
    if free < gb + MARGIN_GB:
        raise DiskSpaceError(f"{what} needs about {gb:.1f} GB (+{MARGIN_GB} GB margin); {free:.1f} GB free")


def build_package(names: Iterable[str], make: Callable[[str], ct.models.MLModel], out: Path, work: Path,
                  batch: int = 2, log: Callable[[str], None] = print) -> dict:
    """Convert every function with `make(name)` and merge them into one multifunction package at `out`, in this
    process. For small models (tests). The stages use `egcm.parts.build`, which isolates each part and merge in its
    own process, because the real model does not fit this Mac's memory when built in one process.

    Returns {"parts_mb": {name: size of its part}, "package_mb": size of `out`}."""
    work.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    done: list[str] = []
    pending: list[Path] = []
    parts_mb: dict[str, float] = {}

    def flush() -> None:
        pkg_gb = size_mb(out) / 1e3 if out.exists() else 0.0
        _need(work, 2 * (pkg_gb + sum(size_mb(p) for p in pending) / 1e3), "merge")
        sources = [(out, fn) for fn in done] + [(p, "main") for p in pending]
        _merge(sources, out)
        done.extend(p.stem for p in pending)
        for p in pending:
            shutil.rmtree(p)
        pending.clear()
        log(f"merged {len(done)} functions: {size_mb(out):.0f} MB")

    for name in names:
        part = work / f"{name}.mlpackage"
        shutil.rmtree(part, ignore_errors=True)
        last = max(parts_mb.values(), default=0.0) / 1e3
        _need(work, 2 * last, f"converting {name}")  # the part plus coremltools' temporary copy
        m = make(name)
        m.save(str(part))
        del m
        parts_mb[name] = size_mb(part)
        log(f"{name}: part {parts_mb[name]:.0f} MB")
        pending.append(part)
        if len(pending) == batch:
            flush()
    if pending:
        flush()
    shutil.rmtree(work, ignore_errors=True)
    return {"parts_mb": parts_mb, "package_mb": size_mb(out)}


def size_mb(path: Path) -> float:
    """On-disk size of a file or directory in MB (1e6 bytes)."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def compile_package(pkg: Path, dest: Path) -> Path:
    """Compile `pkg` to `dest` (.mlmodelc), replacing it."""
    shutil.rmtree(dest, ignore_errors=True)
    compiled = Path(ct.utils.compile_model(str(pkg)))
    shutil.move(str(compiled), str(dest))
    return dest


def _array(feature) -> tuple[str, str, list[int]]:
    a = feature.type.multiArrayType
    return feature.name, ct.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Name(a.dataType), list(a.shape)


def contract_problems(pkg: Path, names: Iterable[str]) -> list[str]:
    """Differences between each function's interface in `pkg` and the contract (empty means it matches)."""
    want_in = {"input_ids": "INT32", "attention_mask": "INT32", "pixels": "FLOAT32"}
    spec = ct.utils.load_spec(str(pkg))
    funcs = {f.name: f for f in spec.description.functions}
    problems = []
    for name in names:
        if name not in funcs:
            problems.append(f"{name}: missing")
            continue
        f = funcs[name]
        got_in = {i.name: _array(i)[1:] for i in f.input}
        expect = {s.name: (want_in[s.name], list(s.shape.shape)) for s in input_specs(name)}
        if got_in != expect:
            problems.append(f"{name}: inputs {got_in} != {expect}")
        outs = [_array(o) for o in f.output]
        if outs != [("embedding", "FLOAT32", [1, EMBEDDING_DIM])]:
            problems.append(f"{name}: outputs {outs}")
    if spec.description.defaultFunctionName != DEFAULT_FUNCTION:
        problems.append(f"default function is {spec.description.defaultFunctionName}")
    return problems


def disk_free_gb(path: Path) -> float:
    out = subprocess.run(["df", "-k", str(path)], capture_output=True, text=True, check=True).stdout.splitlines()[-1]
    return int(out.split()[3]) / 1e6
