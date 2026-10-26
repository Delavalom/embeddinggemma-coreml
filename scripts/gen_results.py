"""Write docs/RESULTS.md (parity, every s4 variant, s5 pixels, device benchmarks) from the stage reports.

Run as `python -m scripts.gen_results` (or `just results`). Reads artifacts/*/report.json and the committed
docs/results/*.json (s6 per device). GENERATED: never edit docs/RESULTS.md by hand; s7 publishes it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from egcm.gates import op_for
from egcm.paths import HF_REPO, HF_REVISION, ROOT, STAGES
from egcm.report import Report
from scripts.gen_status import fmt, key_metrics, load, load_results, stale

# The s4 variant metrics (egcm.ranking.metrics, ADR-0007): ranking against the bucketed FP reference, bucketing
# fidelity against exact-aspect. `vs_raw` recall@10 is recorded only.
S4_COLUMNS = (("download_mb", "download MB"), ("package_mb", "package MB"), ("min_cosine", "min cosine"),
              ("min_recall_at_10", "recall@10"), ("min_p5_spearman_top50", "p5 Spearman top-50"),
              ("min_t2i_top1_in_top10", "exact-aspect top-1 in top 10"),
              ("max_i2i_label_p10_drop", "i2i p@10 drop"), ("ane_fraction_min", "min ANE fraction"))


def _cell(v: object) -> str:
    return "-" if v is None else fmt(v)


def gate_rows(stage: str, r: Report, is_old: bool) -> list[str]:
    failed = {f.split(":", 1)[0] for f in r.failures}
    status = f"{r.status}{' (stale)' if is_old else ''}"
    rows = []
    for k, g in (r.gate_values or {}).items():
        if r.status == "passed" and r.passed:
            ok = "yes"
        elif r.status == "failed":
            ok = "no" if k in failed else "yes"
        else:  # error or running: the metrics were never evaluated
            ok = "-"
        rows.append(f"| {stage} | {status} | {k} | {_cell(r.metrics.get(k))} "
                    f"| {op_for(k, g)} {fmt(g)} | {ok} |")
    if not rows:
        rows.append(f"| {stage} | {status} | - | - | - | {'; '.join(r.failures) or '-'} |")
    return rows


def parity_section(reports: dict, root: Path) -> list[str]:
    lines = ["## Gates", "", "| stage | status | metric | value | gate | ok |", "|---|---|---|---|---|---|"]
    for stage in STAGES:
        r = reports.get(stage)
        if r is None:
            lines.append(f"| {stage} | not run | - | - | - | - |")
        elif isinstance(r, str):
            lines.append(f"| {stage} | invalid | - | - | - | {r} |")
        else:
            lines += gate_rows(stage, r, stale(r, root))
    return lines


def s4_section(r: Report | str | None) -> list[str]:
    if not isinstance(r, Report) or not r.details.get("variants"):
        return []
    variants = r.details["variants"]
    limit = r.gate_values.get("max_download_mb")
    lines = ["", "## s4 variants", "", f"Chosen: `{r.details.get('chosen', '-')}` (winner: "
             f"`{r.details.get('winner') or 'none'}`). Variants are tried smallest first; the first that passes "
             "every gate wins, so later ones may not have run."
             + (f" Download limit: {fmt(limit)} MB." if limit is not None else ""), "",
             "| variant | " + " | ".join(h for _, h in S4_COLUMNS)
             + " | recall@10 vs exact-aspect | passes | description |",
             "|---|" + "---|" * (len(S4_COLUMNS) + 3)]
    for name, m in variants.items():
        passes = m.get("passes", {})
        failed = [k for k, ok in passes.items() if not ok]
        verdict = "all" if passes and not failed else ("fails " + ", ".join(failed) if failed else "-")
        lines.append(f"| {name} | " + " | ".join(_cell(m.get(k)) for k, _ in S4_COLUMNS)
                     + f" | {_cell(m.get('vs_raw', {}).get('min_recall_at_10'))} | {verdict} "
                     f"| {m.get('description', '-')} |")
    return lines


def s5_section(r: Report | str | None) -> list[str]:
    if not isinstance(r, Report) or "pixels" not in r.details:
        return []
    px, cos = r.details["pixels"], r.details.get("cosine", {})
    lines = ["", "## s5 Swift parity", "", "Pixels (0-1 units; PNG and synthetic are gated, JPEG includes the decoder "
             "and is recorded only, ADR-0004):", "", "| inputs | p99 | max | bit-exact |", "|---|---|---|---|"]
    for g in ("png", "synthetic", "jpeg"):
        lines.append(f"| {g} | {_cell(px.get(f'{g}_pixel_p99'))} | {_cell(px.get(f'{g}_pixel_max'))} "
                     f"| {px.get('bit_exact', {}).get(g, '-')} |")
    lines += s5_references(r.details)
    groups = [(f"text {k}", v) for k, v in cos.get("text", {}).items()]
    groups += [(f"image {k}", v) for k, v in cos.get("image", {}).items()]
    if groups:
        lines += ["", "Against s0 (recorded only; on JPEGs it includes the decoder, ADR-0008):", "",
                  "| vectors | n | min cosine | mean cosine |", "|---|---|---|---|"]
        lines += [f"| {k} | {v.get('n', '-')} | {_cell(v.get('min'))} | {_cell(v.get('mean'))} |" for k, v in groups]
    return lines


def s5_references(d: dict) -> list[str]:
    """ADR-0008: one row per reference. Runtime and the FP reference on Swift's pixels are gated; s0 and the
    decoder are recorded."""
    rows = []
    groups = (d.get("runtime") or {}).get("by_group") or {}
    if groups:
        low = min(g["min"] for g in groups.values())
        rows.append(("Swift vs Python Core ML, same inputs (gated)", low, None, None))
    for key, label in (("vs_fp_on_swift_pixels", "Swift vs FP reference on Swift's pixels (gated)"),
                       ("vs_s0", "Swift vs s0 (recorded)"),
                       ("decoder", "FP on Swift's pixels vs s0: the decoder (recorded)")):
        if v := d.get(key):
            rows.append((label, v.get("min_cosine", v.get("min_cosine_image")), v.get("min_recall_at_10"),
                         v.get("min_p5_spearman_top50")))
    if not rows:
        return []
    out = ["", "| comparison | min cosine | recall@10 (worst budget) | p5 Spearman top-50 |", "|---|---|---|---|"]
    return out + [f"| {k} | {_cell(c)} | {_cell(r)} | {_cell(s)} |" for k, c, r, s in rows]


def device_section(reports: dict, results: dict) -> list[str]:
    rows = [("artifacts/s6_device/report.json", reports.get("s6_device"))] if "s6_device" in reports else []
    rows += [(f"docs/results/{n}", r) for n, r in results.items()
             if isinstance(r, str) or r.stage == "s6_device"]
    if not rows:
        return []
    lines = ["", "## Device benchmarks (s6)", "", "| source | device | status | key metrics |", "|---|---|---|---|"]
    for src, r in rows:
        if isinstance(r, str):
            lines.append(f"| {src} | - | invalid | {r} |")
        else:
            device = r.details.get("device") or ", ".join(r.details.get("devices") or {}) or "-"  # s6: per model
            lines.append(f"| {src} | {device} | {r.status} | {key_metrics(r)} |")
    return lines


def render(root: Path = ROOT) -> str:
    reports, results = load(root), load_results(root)
    lines = ["# Results", "",
             "Generated by `python -m scripts.gen_results` from `artifacts/*/report.json` and `docs/results/*.json`. "
             "Do not edit by hand.", "", f"Model: `{HF_REPO}` at revision `{HF_REVISION}`.", ""]
    lines += parity_section(reports, root)
    lines += s4_section(reports.get("s4_compress"))
    lines += s5_section(reports.get("s5_swift"))
    lines += device_section(reports, results)
    return "\n".join(lines) + "\n"


def main() -> None:
    text = render()
    out = ROOT / "docs" / "RESULTS.md"
    out.write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
