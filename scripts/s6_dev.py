# group: convert
"""s6's measurement without the stage runner: for iterating on the device while s4/s5 are stale or running.

Same bundle, build, compile run and benchmark as `just stage 6`, and the same gate check, but it writes no
report.json (only artifacts/s6_device/dev/), so it never counts as an s6 result. Upstream freshness is not checked:
it bundles whatever s4 shipped last.

    uv run --exact --group convert python -m scripts.s6_dev [--device <UDID>]
"""

from __future__ import annotations

import argparse
import json

from egcm.gates import evaluate, targets
from egcm.paths import stage_dir
from egcm.report import Report
from stages.s6_device import run as s6


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device")
    p.add_argument("--package", help="as for stages.s6_device.run")
    args = p.parse_args()
    out = stage_dir(s6.STAGE) / "dev"
    out.mkdir(parents=True, exist_ok=True)
    expected, _ = s6.bundle(args, out)
    ctl = s6.Devicectl()
    device = s6.pick_device(ctl.list_devices(), args.device)
    result = s6.benchmark(device, s6.build_app(device, out), out, ctl)
    (out / f"{device.product_type}.json").write_text(json.dumps(result, indent=1) + "\n")
    s6.check_revisions(result, expected)
    metrics, _ = s6.summarize([result], targets(s6.STAGE)["target_ms_per_image_70"])
    verdict = evaluate(Report(stage=s6.STAGE, metrics=metrics), s6.STAGE)
    print(json.dumps({"metrics": metrics, "cold_loads": s6.cold_loads(result), "passed": verdict.passed,
                      "failures": verdict.failures}, indent=1))


if __name__ == "__main__":
    main()
