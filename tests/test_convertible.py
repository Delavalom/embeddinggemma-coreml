"""The rewrite's building blocks convert with coremltools (FP16, iOS18, jit.trace) and run on the Neural Engine.

Each case is RMSNormCF -> AttentionCF (with key_mask) as one traced graph, for both RoPE variants.

- Conversion and accuracy use tiny dims (fast).
- The placement check uses the real head layout (text sliding layer: hidden 512, 4 heads, 2 kv heads, head_dim 256;
  vision: hidden 768, head_dim 64, 4 of the 12 heads to keep it fast) at S = 64. Core ML places ops by cost, and
  below about that size it keeps a whole graph on the CPU even though every op is ANE-capable (measured on an
  M1 Pro), so tiny dims say nothing about placement. Inputs and outputs are FP16 so no fp32 -> fp16 cast op sits
  on the CPU. The first ANE compile of a new graph can take much longer on a loaded machine; later runs hit the
  system's compile cache.
"""

import platform

import numpy as np
import pytest
import torch

from egcm.model.attention import AttentionCF
from egcm.model.masks import key_mask
from egcm.model.norm import RMSNormCF
from egcm.model.rope import rope_1d, rope_2d

ct = pytest.importorskip("coremltools")
on_mac = pytest.mark.skipif(platform.system() != "Darwin", reason="Core ML runtime needs macOS")


class Block(torch.nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, S, halves):
        super().__init__()
        self.norm = RMSNormCF(hidden)
        self.attn = AttentionCF(hidden, heads, kv_heads, head_dim, eps=1e-6, rope_halves=halves)
        side = int(S**0.5)
        cos, sin = rope_1d(S, head_dim, 1e4) if halves == 1 else rope_2d(side, S // side, head_dim, 100.0)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x, valid):
        return self.attn(self.norm(x), self.cos, self.sin, key_mask(valid))


def build(hidden, heads, kv_heads, head_dim, S, halves, valid_len):
    torch.manual_seed(0)
    m = Block(hidden, heads, kv_heads, head_dim, S, halves).eval()
    for name, p in m.named_parameters():
        p.data = torch.rand_like(p) + 0.5 if name.endswith("norm.weight") else p.data.normal_(0, 0.05)
    x = torch.randn(1, hidden, 1, S) * 3
    valid = (torch.arange(S) < valid_len).float()[None]
    return m, x, valid


def convert(m, x, valid, io=np.float16):
    traced = torch.jit.trace(m, (x, valid))
    return ct.convert(
        traced,
        inputs=[ct.TensorType("x", shape=x.shape, dtype=io), ct.TensorType("valid", shape=valid.shape, dtype=io)],
        outputs=[ct.TensorType("y", dtype=io)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )


def mil_ops(ml):
    return list(ml._mil_program.functions["main"].operations)


TINY = {"text": (64, 2, 1, 32, 16, 1, 12), "vision": (64, 2, 2, 32, 16, 2, 16)}
REAL = {"text": (512, 4, 2, 256, 64, 1, 50), "vision": (768, 4, 4, 64, 64, 2, 64)}  # vision: 4 of the 12 heads


@pytest.mark.parametrize("case", list(TINY))
def test_converts_and_matches_torch(case):
    m, x, valid = build(*TINY[case])
    ml = convert(m, x, valid, io=np.float32)
    scalar_consts = [
        float(v.val) for op in mil_ops(ml) if op.op_type == "mul" for v in op.inputs.values()
        if v.val is not None and np.size(v.val) == 1
    ]
    assert -1.0 not in scalar_consts  # the rotate_half sign is folded into the sin table
    assert not [op for op in mil_ops(ml) if op.op_type in ("neg", "negative")]
    if platform.system() != "Darwin":
        return
    got = ml.predict({"x": x.numpy(), "valid": valid.numpy()})["y"]
    want = m(x, valid).detach().numpy()
    assert np.isfinite(got).all()
    assert np.abs(got - want).max() <= 2e-2 * np.abs(want).max()  # fp16 compute


@on_mac
@pytest.mark.parametrize("case", list(REAL))
def test_every_op_on_neural_engine(case):
    from coremltools.models.compute_plan import MLComputePlan

    m, x, valid = build(*REAL[case])
    ml = convert(m, x, valid)
    plan = MLComputePlan.load_from_path(ml.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    off_ane = []
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:  # const ops have no device
            continue
        device = type(usage.preferred_compute_device).__name__
        if device != "MLNeuralEngineComputeDevice":
            off_ane.append((op.operator_name, device))
    assert not off_ane, off_ane
