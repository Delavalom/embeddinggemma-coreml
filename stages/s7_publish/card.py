"""The Hugging Face model card: `model_card.md` (a `string.Template`) filled from the stage reports.

Every number in the card comes from artifacts/<stage>/report.json (s4's chosen variant, s6's devices), so the card
cannot drift from what was measured. s7 refuses to publish unless s0-s6 passed and are current, so a published card
has every value; `PENDING` only shows when the card is rendered early (tests, previews).
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from egcm.buckets import BUDGETS
from egcm.manifest import ARCHIVE_NAME, MANIFEST_JSON
from egcm.paths import HF_REPO, HF_REVISION, STAGES
from egcm.report import Report, read_report

TEMPLATE = Path(__file__).with_name("model_card.md")
PENDING = "pending"
GITHUB_REPO = "Delavalom/embeddinggemma-coreml"


def _report(root: Path, stage: str) -> Report | None:
    p = Path(root) / "artifacts" / stage / "report.json"
    try:
        return read_report(p) if p.is_file() else None
    except Exception:
        return None


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _num(v, digits: int) -> str:
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else PENDING


def _cos(v) -> str:
    """A cosine that may be within 1e-6 of 1 (s1, s2): "1 - 1.4e-12" instead of a misleading "1.000000"."""
    if isinstance(v, float) and 1 - v < 1e-6:
        return f"1 - {1 - v:.1e}"
    return _num(v, 6)


def _pct(v, digits: int) -> str:
    return f"{100 * v:.{digits}f}%" if isinstance(v, (int, float)) and not isinstance(v, bool) else PENDING


def _short(s: str | None, n: int) -> str:
    return s[:n] if s else PENDING


def provenance_rows(root: Path) -> str:
    rows = []
    for stage in STAGES[:7]:
        r = _report(root, stage)
        if r is None:
            rows.append(f"| {stage} | not run | {PENDING} | {PENDING} |")
        else:
            rows.append(f"| {stage} | {r.status} | `{_short(r.git_sha, 12)}` | `{_short(r.code_hash, 16)}` |")
    return "\n".join(rows)


def device_rows(s6: Report | None) -> str:
    devices = _get(s6.details, "devices") if s6 else None
    if not devices:
        return f"| {PENDING}: filled from s6_device | - | - | - | - | - |"
    rows = []
    for model, d in sorted(devices.items()):
        ms = {b: _num(_get(d, "budgets", b, "ms_per_image", "median"), 1) for b in ("70", "280")}
        rows.append(f"| {model} | {d.get('os_version', '-')} | {ms['70']} | {ms['280']} "
                    f"| {_num(_get(d, 'query', 'ms', 'median'), 1)} | {_num(d.get('peak_rss_mb'), 0)} |")
    return "\n".join(rows)


def worst_images(variant: dict, n: int = 3) -> str:
    """The `n` lowest image cosines over every image function, e.g. "i161 0.99133 (embed_image_280_landscape)"."""
    worst = [(c, i, f) for f, d in (_get(variant, "by_function", "image") or {}).items()
             for i, c in d.get("worst", [])]
    if not worst:
        return PENDING
    return ", ".join(f"{i} {c:.5f} (`{f}`)" for c, i, f in sorted(worst)[:n])


def budget_values(s4: Report | None, manifests: dict[str, dict] | None) -> dict[str, str]:
    """Per image budget b (ADR-0010): `download_mb_<b>`, `package_mb_<b>` (s4's `details.budgets`) and
    `unpacked_mb_<b>` (the uploaded b<b>/manifest.json)."""
    out = {}
    for b in BUDGETS:
        shipped = _get(s4.details, "budgets", str(b)) if s4 else None
        unpacked = _get(manifests or {}, str(b), "unpacked_size")
        out |= {f"download_mb_{b}": _num(_get(shipped, "download_mb"), 1),
                f"package_mb_{b}": _num(_get(shipped, "package_mb"), 1),
                f"unpacked_mb_{b}": _num(unpacked / 1e6 if unpacked else None, 1)}
    return out


def values(root: Path, manifests: dict[str, dict] | None = None) -> dict[str, str]:
    """Every `$name` of model_card.md except `repo` and `results`. `manifests`: image budget ("70", "280") -> the
    uploaded b<budget>/manifest.json (egcm.manifest.write), for the unpacked sizes."""
    s0, s1, s2, s3, s4, s5, s6 = (_report(root, s) for s in STAGES[:7])
    m = lambda r, k: r.metrics.get(k) if r else None  # noqa: E731
    det = lambda r, *k: _get(r.details, *k) if r else None  # noqa: E731
    chosen = det(s4, "chosen")
    v = det(s4, "variants", chosen) if chosen else None
    v = v if isinstance(v, dict) else {}
    return budget_values(s4, manifests) | {
        "github": GITHUB_REPO,
        "base_repo": HF_REPO,
        "base_revision": det(s0, "revision") or HF_REVISION,
        "base_sha256": det(s0, "model_sha256") or PENDING,
        "base_license": det(s0, "license") or PENDING,
        "transformers": det(s0, "versions", "transformers") or PENDING,
        "sentence_transformers": det(s0, "versions", "sentence_transformers") or PENDING,
        "coremltools": det(s2, "coremltools") or PENDING,
        "s1_text": _cos(m(s1, "min_cosine_text")), "s1_image": _cos(m(s1, "min_cosine_image")),
        "s2_text": _cos(m(s2, "min_cosine_text")), "s2_image": _cos(m(s2, "min_cosine_image")),
        "s3_text": _num(m(s3, "min_cosine_text"), 5), "s3_image": _num(m(s3, "min_cosine_image"), 5),
        "s3_p1_image": _num(m(s3, "min_p1_cosine_image"), 5), "s3_ane": _pct(m(s3, "min_ane_fraction"), 2),
        "raw_vs_bucket_70": _num(det(s0, "raw_vs_bucket_cosine", "70", "min"), 3),
        "raw_vs_bucket_280": _num(det(s0, "raw_vs_bucket_cosine", "280", "min"), 3),
        "worst_images": worst_images(v),
        "variant": chosen or PENDING,
        "variant_description": v.get("description", PENDING),
        "download_mb": _num(m(s4, "max_download_mb") if det(s4, "budgets") else None, 1),
        "archive": ARCHIVE_NAME, "manifest": MANIFEST_JSON,
        "max_download_mb": _num(s4.gate_values.get("max_download_mb") if s4 else None, 0),
        "text_cos": _num(v.get("min_cosine_text"), 5), "image_cos": _num(v.get("min_cosine_image"), 5),
        "p1_image_cos": _num(v.get("min_p1_cosine_image"), 5),
        "recall_70": _num(_get(v, "vs_bucket", "by_budget", "70", "recall_at_10"), 3),
        "recall_280": _num(_get(v, "vs_bucket", "by_budget", "280", "recall_at_10"), 3),
        "spearman_70": _num(_get(v, "vs_bucket", "by_budget", "70", "p5_spearman_top50"), 3),
        "spearman_280": _num(_get(v, "vs_bucket", "by_budget", "280", "p5_spearman_top50"), 3),
        "raw_recall_70": _num(_get(v, "vs_raw", "by_budget", "70", "recall_at_10"), 3),
        "raw_recall_280": _num(_get(v, "vs_raw", "by_budget", "280", "recall_at_10"), 3),
        "top1_in_top10": _pct(v.get("min_t2i_top1_in_top10"), 1),
        "i2i_drop": _num(v.get("max_i2i_label_p10_drop"), 3),
        "ms_text_16": _num(_get(v, "latency_cpu_and_ne", "embed_text_16", "median_ms"), 1),
        "ms_image_70_square": _num(_get(v, "latency_cpu_and_ne", "embed_image_70_square", "median_ms"), 1),
        "ane_fraction": _pct(v.get("ane_fraction_min"), 2),
        "s5_cos": _num(m(s5, "min_cosine"), 5), "s5_pixel": _num(m(s5, "max_pixel_abs_diff"), 6),
        "s5_runtime": _cos(m(s5, "min_cosine_runtime")),
        "s5_tokens": str(m(s5, "token_ids_identical")) if s5 else PENDING,
        "s6_ms_70": _num(m(s6, "ms_per_image_70_median"), 1), "s6_ms_280": _num(m(s6, "ms_per_image_280_median"), 1),
        "s6_ms_query": _num(m(s6, "ms_per_query_median"), 1), "s6_rss": _num(m(s6, "max_peak_rss_mb"), 0),
        "s6_devices": device_rows(s6),
        "provenance": provenance_rows(root),
    }


def render(repo: str, results_md: str, vals: dict[str, str]) -> str:
    """The card. `substitute` (not safe_substitute): a placeholder without a value is a bug, so it raises."""
    return Template(TEMPLATE.read_text()).substitute(vals, repo=repo, results=results_md.strip())
