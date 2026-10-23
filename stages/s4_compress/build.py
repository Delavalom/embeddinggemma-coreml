"""Build every s4 variant in one conversion pass: each function is converted to FP16 once, every recipe compresses
that model in memory (coremltools rebuilds a fresh program from the spec per call, so the FP16 model is unchanged),
and each compressed part goes to its own growing multifunction package.

Disk stays small: the compressed parts are 0.1 to 0.5 GB, and the FP16 model only lives in coremltools' temporary
package while its function is being processed.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

import coremltools as ct

from egcm.convert import _merge, _need, size_mb


class Builder:
    """One growing multifunction package: `add(name, mlmodel)` saves a part; every `batch` parts are merged."""

    def __init__(self, out: Path, work: Path, batch: int = 2, log: Callable[[str], None] = print):
        self.out, self.work, self.batch, self.log = out, work, batch, log
        work.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(out, ignore_errors=True)
        self.done: list[str] = []
        self.pending: list[Path] = []
        self.parts_mb: dict[str, float] = {}

    def add(self, name: str, m: ct.models.MLModel) -> None:
        part = self.work / f"{name}.mlpackage"
        shutil.rmtree(part, ignore_errors=True)
        m.save(str(part))
        self.parts_mb[name] = size_mb(part)
        self.pending.append(part)
        if len(self.pending) == self.batch:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        pkg_gb = size_mb(self.out) / 1e3 if self.out.exists() else 0.0
        _need(self.work, 2 * (pkg_gb + sum(size_mb(p) for p in self.pending) / 1e3), f"merge {self.out.name}")
        _merge([(self.out, fn) for fn in self.done] + [(p, "main") for p in self.pending], self.out)
        self.done.extend(p.stem for p in self.pending)
        for p in self.pending:
            shutil.rmtree(p)
        self.pending.clear()
        self.log(f"{self.out.name}: {len(self.done)} functions, {size_mb(self.out):.0f} MB")

    def finish(self) -> dict:
        self.flush()
        shutil.rmtree(self.work, ignore_errors=True)
        return {"parts_mb": self.parts_mb, "package_mb": size_mb(self.out)}


def build_variants(names: Iterable[str], make_fp16: Callable[[str], ct.models.MLModel],
                   recipes: dict[str, Callable[[ct.models.MLModel], ct.models.MLModel]], out: Path,
                   log: Callable[[str], None] = print) -> dict[str, dict]:
    """{recipe: sizes}; the packages are `out / f"{recipe}.mlpackage"`."""
    builders = {r: Builder(out / f"{r}.mlpackage", out / "parts" / r, log=log) for r in recipes}
    for name in names:
        _need(out, 2.5, f"converting {name}")  # FP16 model (temporary package) + compressed parts
        fp16 = make_fp16(name)
        for r, compress in recipes.items():
            builders[r].add(name, compress(fp16))
        del fp16
        log(f"{name}: " + ", ".join(f"{r} {b.parts_mb[name]:.0f} MB" for r, b in builders.items()))
    sizes = {r: b.finish() for r, b in builders.items()}
    shutil.rmtree(out / "parts", ignore_errors=True)
    return sizes
