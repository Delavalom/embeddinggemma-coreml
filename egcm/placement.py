"""Neural Engine placement of every function in a compiled package, from MLComputePlan (s3, s4)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan


def ane_placement(compiled: Path, compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE) -> dict[str, dict]:
    """Per function: ops with a device (const ops have none), how many prefer the Neural Engine, the fraction, and
    the ops that do not ("op_type@Device": count), from MLComputePlan of the compiled package."""
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
        out[name] = {"ops": n, "ane_ops": ane, "fraction": ane / n if n else 0.0, "off_ane": dict(off)}
    return out
