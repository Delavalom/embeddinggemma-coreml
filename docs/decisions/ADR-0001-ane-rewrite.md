# ADR-0001: Rewrite the model for the Neural Engine instead of tracing Hugging Face's code

**Status:** accepted, 2026-10-23

## Context

google/embeddinggemma-2 ships as Hugging Face Transformers code (`EmbeddingGemma2Model`, the Gemma 4 vision tower).
Tracing that code into Core ML gives a graph in the layout PyTorch uses on GPUs: hidden states `(B, S, C)`,
`nn.Linear`, attention through 4-D `(B, H, S, D)` tensors with transposes, masks built at run time from shapes, and
RMSNorms that square raw activations. Apple's guidance for transformers on the Neural Engine
([ml-ane-transformers](https://github.com/apple/ml-ane-transformers), "Deploying Transformers on the Apple Neural
Engine", 2022) is that this layout makes the Neural Engine fall back to the CPU or copy memory around, and that a
model written in the Neural Engine's native layout runs much faster at the same accuracy:

- channels-first `(B, C, 1, S)` tensors, so the last axis is the sequence and is 64-byte aligned;
- every linear as a 1x1 `Conv2d`;
- attention split per head, with `einsum` on 4-D tensors and no 5-D reshapes;
- no `(B, S, C)` <-> `(B, C, 1, S)` transposes inside the model.

The model card also says BF16 or FP32, not FP16, and the Neural Engine computes in FP16. So the rewrite also has to
be FP16-safe by construction where it can (the per-token dynamic RMSNorm, `egcm/model/norm.py`).

## Decision

Write the two towers again in plain PyTorch in the Neural Engine layout (`egcm/model/`), load Google's weights into
it with an explicit, strict mapping (`egcm/weights.py`), and convert that rewrite, never the Hugging Face code, with
coremltools (`egcm/convert.py`). Every fixed shape (text lengths 16/32/64/128, ten image buckets) is its own function
of one multifunction `.mlpackage`, with the weights stored once. Every size, RoPE table and position embedding is a
constant fixed when the function is built, so the traced graph has no shape arithmetic.

Correctness is proved in stages, each against the official sentence-transformers vectors from s0: the rewrite in
PyTorch FP32 (s1, cosine ≥ 0.9999), the Core ML conversion at FP32 on the CPU (s2, ≥ 0.9999), then FP16 on the
Neural Engine (s3, ≥ 0.995 with ≥ 95% of ops on the Neural Engine).

## Consequences

- We maintain a second implementation of the model. `tests/upstream.py` and `tests/upstream_towers.py` (copies of
  the upstream maths) and s1's `debug.npz` comparison catch drift; a new upstream revision means rerunning s0 and s1.
- Departures from upstream are exact in maths and documented where they are made: the dynamic RMSNorm, the RoPE
  sign folded into the sin table, the pooler's `sqrt(768)` dropped before the weightless RMSNorm, the patch
  projection as a stride-16 conv.
- `transformers` is never imported by the convert stages; only s0 uses it, in its own environment.
