"""Streaming multifunction merge: the same result as coremltools' `save_multifunction`, at constant memory.

`save_multifunction` loads every function's program with all of its constants before deduplicating them, so merging
the package so far plus one part needs memory for every function's weights (measured: 8.1 GB at 5 FP32 functions,
growing about 1.1 GB per function). Here the merge works on the protobuf specs and the weight files directly:

- each source function's spec is copied into the merged ML program (renamed to its target name);
- every weight it references (a `Value` with a `blobFileValue`) is read from the source's `weight.bin` with
  coremltools' own blob reader, written to the merged `weight.bin` with its blob writer the first time its
  (dtype, content) is seen, and the reference is pointed at the merged offset. A weight shared by many functions
  is stored once, as `save_multifunction` does.

Memory is the specs plus one weight at a time (the largest is the 262,144 x 512 table). The result loads, compiles
and predicts like a `save_multifunction` package (tests/test_mfmerge.py compares them).
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterator
from pathlib import Path

from coremltools.libmilstoragepython import _BlobStorageReader, _BlobStorageWriter
from coremltools.proto import MIL_pb2, Model_pb2

WEIGHT_FILE = "@model_path/weights/weight.bin"
_IO = {"FLOAT16": "fp16", "FLOAT32": "float", "INT8": "int8", "UINT8": "uint8", "INT16": "int16", "UINT16": "uint16",
       "INT32": "int32", "UINT32": "uint32", "INT4": "int4", "UINT4": "uint4", "UINT1": "uint1", "UINT2": "uint2",
       "UINT3": "uint3", "UINT6": "uint6"}
_VALUE = MIL_pb2.Value.DESCRIPTOR.full_name


def _spec_path(pkg: Path) -> Path:
    return Path(pkg) / "Data" / "com.apple.CoreML" / "model.mlmodel"


def _weights_path(pkg: Path) -> Path:
    return Path(pkg) / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"


def load_spec(pkg: Path) -> Model_pb2.Model:
    spec = Model_pb2.Model()
    spec.ParseFromString(_spec_path(pkg).read_bytes())
    return spec


def blob_values(msg) -> Iterator[MIL_pb2.Value]:
    """Every `Value` with a `blobFileValue` anywhere inside `msg` (op attributes, inline arguments, ...)."""
    if msg.DESCRIPTOR.full_name == _VALUE and msg.HasField("blobFileValue"):
        yield msg
        return
    for field, value in msg.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        if field.message_type.GetOptions().map_entry:
            items = value.values()
            if field.message_type.fields_by_name["value"].type != field.TYPE_MESSAGE:
                continue
        elif hasattr(value, "ListFields"):  # a single message
            items = [value]
        else:  # a repeated field
            items = value
        for item in items:
            yield from blob_values(item)


def _function_description(spec: Model_pb2.Model, name: str) -> Model_pb2.FunctionDescription:
    for f in spec.description.functions:
        if f.name == name:
            return f
    fd = Model_pb2.FunctionDescription(name=name)  # a single-function model keeps its interface at the top level
    fd.input.extend(spec.description.input)
    fd.output.extend(spec.description.output)
    fd.state.extend(spec.description.state)
    fd.predictedFeatureName = spec.description.predictedFeatureName
    fd.predictedProbabilitiesName = spec.description.predictedProbabilitiesName
    return fd


def merge(sources: list[tuple[Path, str, str]], out: Path, default: str) -> dict:
    """Write `out` with one function per (package, source function, target name). Returns blob counts."""
    tmp = out.with_name(out.stem + ".merging.mlpackage")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "Data" / "com.apple.CoreML" / "weights").mkdir(parents=True)
    shutil.copy(Path(sources[0][0]) / "Manifest.json", tmp / "Manifest.json")
    merged = Model_pb2.Model()
    merged.mlProgram.version = 1
    writer = _BlobStorageWriter(str(_weights_path(tmp)), truncate_file=True)
    seen: dict[tuple[str, int, str], int] = {}
    refs = 0
    for pkg, src, target in sources:
        spec = load_spec(pkg)
        if target in merged.mlProgram.functions:
            raise ValueError(f"duplicate function {target}")
        merged.specificationVersion = max(merged.specificationVersion, spec.specificationVersion)
        if not merged.description.HasField("metadata") and spec.description.HasField("metadata"):
            merged.description.metadata.CopyFrom(spec.description.metadata)
        fn = merged.mlProgram.functions[target]
        fn.CopyFrom(spec.mlProgram.functions[src])
        reader = _BlobStorageReader(str(_weights_path(pkg)))
        for v in blob_values(fn):
            dtype = MIL_pb2.DataType.Name(v.type.tensorType.dataType)
            arr = getattr(reader, f"read_{_IO[dtype]}_data")(v.blobFileValue.offset)
            key = (dtype, arr.size, hashlib.sha256(arr.tobytes()).hexdigest())
            if key not in seen:
                seen[key] = getattr(writer, f"write_{_IO[dtype]}_data")(arr)
            v.blobFileValue.fileName = WEIGHT_FILE
            v.blobFileValue.offset = seen[key]
            refs += 1
            del arr
        del reader
        merged.description.functions.append(_function_description(spec, src))
        merged.description.functions[-1].name = target
    del writer  # flushes and closes weight.bin
    names = [t for _, _, t in sources]
    merged.description.defaultFunctionName = default if default in names else names[0]
    _spec_path(tmp).write_bytes(merged.SerializeToString())
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return {"functions": names, "blob_refs": refs, "blobs_written": len(seen)}
