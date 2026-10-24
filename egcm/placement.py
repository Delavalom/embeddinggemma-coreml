"""Neural Engine placement of every function of a package, from MLComputePlan (s3, s4).

`MLComputePlan.load_from_path` takes no function name: on a multifunction model it plans the default function only
and reports no device for the ops of every other function (measured 2026-10-23: s3 got placement for
`embed_text_16` and 0 ops for the other 13). It also rejects any package that uses the multifunction description
syntax. So `ane_placement_per_function` extracts each function into a plain single-function package
(egcm.mfmerge.extract), compiles and plans it, and deletes it again.
"""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan


def ane_placement(compiled: Path, compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE) -> dict[str, dict]:
    """Per function with planned ops: ops with a device (const ops have none), how many prefer the Neural Engine,
    the fraction, and the ops that do not ("op_type@Device": count). Functions with no planned op are omitted."""
    plan = MLComputePlan.load_from_path(str(compiled), compute_units=compute_units)
    out = {}
    for name, f in plan.model_structure.program.functions.items():
        n, ane, off = 0, 0, Counter()
        for op in f.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue
            n += 1
            device = type(usage.preferred_compute_device).__name__
            if device == "MLNeuralEngineComputeDevice":
                ane += 1
            else:
                off[f"{op.operator_name}@{device.removeprefix('ML').removesuffix('ComputeDevice')}"] += 1
        if n:
            out[name] = {"ops": n, "ane_ops": ane, "fraction": ane / n, "off_ane": dict(off)}
    return out


def ane_placement_per_function(pkg: Path, names: Iterable[str], work: Path,
                               compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE,
                               log: Callable[[str], None] = print) -> dict[str, dict]:
    """`ane_placement` of each function of the (uncompiled) multifunction package `pkg`, one at a time."""
    from egcm import mfmerge
    from egcm.convert import compile_package
    from egcm.e5cache import bundles

    out = {}
    work.mkdir(parents=True, exist_ok=True)
    before = bundles()
    for name in names:
        one, compiled = work / f"{name}.mlpackage", work / f"{name}.mlmodelc"
        mfmerge.extract(pkg, name, one)
        compile_package(one, compiled)
        res = ane_placement(compiled, compute_units)
        if "main" not in res:
            raise RuntimeError(f"MLComputePlan returned no placement for {name}")
        out[name] = res["main"]
        shutil.rmtree(one, ignore_errors=True)
        shutil.rmtree(compiled, ignore_errors=True)
        for b in bundles() - before:  # the plan compiles for the Neural Engine and caches a bundle per function
            shutil.rmtree(b, ignore_errors=True)
        log(f"placement {name}: {out[name]['ane_ops']}/{out[name]['ops']} on the Neural Engine, "
            f"off: {out[name]['off_ane']}")
    shutil.rmtree(work, ignore_errors=True)
    return out
