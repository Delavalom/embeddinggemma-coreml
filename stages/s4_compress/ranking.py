"""s4 metrics for one variant: cosines against s0, ranking of the 120 app queries over the images, download size.

Ranking follows spec section 5 and ADR-0002: the reference is the official model on the *exact-aspect* images (s0
`emb_raw`) with s0's query vectors, the new side is our package on the bucketed images with our query vectors, so
the gate includes the cost of bucketing. The same metrics against s0's bucketed vectors (`emb_bucket`) are recorded
as `vs_bucket` (compression cost alone) and are not gated.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

from egcm.buckets import BUDGETS
from egcm.fixtures import load_texts
from egcm.metrics import recall_at_k, spearman_topk


def query_rows(order: list[str]) -> list[int]:
    kinds = {t["id"]: t["kind"] for t in load_texts()}
    return [i for i, k in enumerate(order) if kinds[k] == "query"]


def metrics(texts: dict, images: dict) -> dict:
    """texts / images as returned by egcm.evaluate.run_texts / run_images."""
    q = query_rows(texts["order"])
    q_ref, q_new = texts["ref"][q], texts["emb"][q]
    out: dict = {"min_cosine_text": float(texts["cos"].min()),
                 "min_cosine_image": min(float(images[b]["cos"].min()) for b in BUDGETS)}
    out["min_cosine"] = min(out["min_cosine_text"], out["min_cosine_image"])
    for ref_key, tag in (("ref_raw", "vs_raw"), ("ref_bucket", "vs_bucket")):
        per = {}
        for b in BUDGETS:
            d_ref, d_new = images[b][ref_key], images[b]["emb"]
            per[b] = {"recall_at_10": recall_at_k(ref=(q_ref, d_ref), new=(q_new, d_new), k=10),
                      "p5_spearman_top50": spearman_topk(ref=(q_ref, d_ref), new=(q_new, d_new), k=50)}
        out[tag] = {"by_budget": per, "min_recall_at_10": min(p["recall_at_10"] for p in per.values()),
                    "min_p5_spearman_top50": min(p["p5_spearman_top50"] for p in per.values())}
    out["min_recall_at_10"] = out["vs_raw"]["min_recall_at_10"]
    out["min_p5_spearman_top50"] = out["vs_raw"]["min_p5_spearman_top50"]
    return out


def zip_package(pkg: Path, dest: Path) -> Path:
    """`dest` (.zip, deflated) holding `pkg` as its top-level directory, the form the app downloads."""
    dest.unlink(missing_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for p in sorted(pkg.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(pkg.parent).as_posix())
    return dest


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checksums(pkg: Path, zipped: Path) -> dict:
    files = {p.relative_to(pkg.parent).as_posix(): sha256(p) for p in sorted(pkg.rglob("*")) if p.is_file()}
    return {"zip": {zipped.name: sha256(zipped), "bytes": zipped.stat().st_size}, "files": files}


def remove(*paths: Path) -> None:
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
