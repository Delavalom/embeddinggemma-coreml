# group: convert
"""s6_device: PhotoSearchDemo's benchmark on a real iPhone.

Gates (gates.toml [s6_device]): max_peak_rss_mb (maximum over devices), min_devices (devices with a result).
The speed target ([s6_device.targets]) is recorded next to the measurement, never gated.
See stages/s6_device/README.md.

    just stage 6                                                          # first physical iPhone
    uv run --exact --group convert python -m stages.s6_device.run --device <UDID>   # a given one
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from egcm import published
from egcm.gates import targets
from egcm.paths import ROOT, stage_dir
from egcm.report import Report
from egcm.stage import run_stage
from stages.s6_device.device import BUNDLE_ID, Device, Devicectl, pick_device, poll_result
from stages.s6_device.team import ensure_team_xcconfig

STAGE = "s6_device"
DEMO = ROOT / "Examples" / "PhotoSearchDemo"
# The published folder (egcm.published) the app bundles as its `Model` resource. Gitignored; written by this stage.
MODEL_DIR = DEMO / "Model"


def log(msg: str) -> None:
    print(f"[s6] {msg}", flush=True)


def xcodegen() -> list[str]:
    return ["xcodegen"] if shutil.which("xcodegen") else ["mise", "exec", "--", "xcodegen"]


def build_command(udid: str, derived: Path) -> list[str]:
    """Release, so the Swift preprocessing in the timed path is optimized."""
    return ["xcodebuild", "-project", str(DEMO / "PhotoSearchDemo.xcodeproj"), "-scheme", "PhotoSearchDemo",
            "-configuration", "Release", "-destination", f"id={udid}", "-derivedDataPath", str(derived),
            "-allowProvisioningUpdates", "build"]


def _check(cmd: list[str], cwd: Path, logfile: Path) -> None:
    with open(logfile, "w") as f:
        rc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"{cmd[0]} failed ({rc}); see {logfile}")


def build_app(device: Device, out: Path) -> Path:
    ensure_team_xcconfig()
    _check([*xcodegen(), "generate", "--spec", "project.yml"], DEMO, out / "xcodegen.log")
    derived = out / "DerivedData"
    app = derived / "Build" / "Products" / "Release-iphoneos" / "PhotoSearchDemo.app"
    # Xcode does not re-sign when only the bundled Model/ changed, so the old app goes first (relinking is cheap).
    shutil.rmtree(app, ignore_errors=True)
    log(f"building PhotoSearchDemo for {device.marketing_name} (log: {out / 'xcodebuild.log'})")
    _check(build_command(device.udid, derived), DEMO, out / "xcodebuild.log")
    if not (app / "Model" / published.FILES_JSON).is_file():
        raise RuntimeError(f"{app} has no Model/{published.FILES_JSON}; the bundling script did not run")
    _check(["codesign", "--verify", "--deep", "--strict", str(app)], DEMO, out / "codesign.log")
    return app


def benchmark(device: Device, app: Path, out: Path, ctl: Devicectl | None = None) -> dict:
    """Installs, launches with `-benchmark`, and waits for this run's result file."""
    ctl = ctl or Devicectl()
    run_id = uuid.uuid4().hex
    log(f"installing on {device.marketing_name} ({device.product_type}, iOS {device.os_version})")
    ctl.install(device, app)
    console = out / f"console_{device.product_type}.log"
    with open(console, "w") as f:
        proc = subprocess.Popen(ctl.launch_command(device, run_id), stdout=f, stderr=subprocess.STDOUT)
        log("launched. Tap Allow Full Access on the phone if it asks for Photos (first run only), "
            f"and keep it unlocked. Console: {console}")
        try:
            return poll_result(lambda: ctl.fetch_result(device, out / "result.json"), run_id,
                               app_exited=proc.poll, log=log)
        finally:
            if proc.poll() is None:
                proc.terminate()


def _worst(results: list[dict], get) -> float | None:
    vals = [v for r in results if (v := get(r)) is not None]
    return max(vals) if vals else None


def summarize(results: list[dict], target_ms_70: float) -> tuple[dict, dict]:
    """(metrics, details) over one result per device. Speed numbers are the worst device's medians."""
    def ms(budget):
        return lambda r: r.get("budgets", {}).get(budget, {}).get("ms_per_image", {}).get("median")

    metrics = {
        "min_devices": len(results),
        "max_peak_rss_mb": _worst(results, lambda r: r.get("peak_rss_mb")),
        "ms_per_image_70_median": _worst(results, ms("70")),
        "ms_per_image_280_median": _worst(results, ms("280")),
        "ms_per_query_median": _worst(results, lambda r: r.get("query", {}).get("ms", {}).get("median")),
    }
    details = {
        "target_ms_per_image_70": target_ms_70,
        "meets_target_70": {r["device_model"]: (m := ms("70")(r)) is not None and m <= target_ms_70 for r in results},
        "devices": {r["device_model"]: r for r in results},
    }
    return metrics, details


def make_fill(args: argparse.Namespace):
    def fill(report: Report) -> None:
        out = stage_dir(STAGE)
        if args.package is None:  # what s4's report says it shipped
            s4 = published.shipped()
            package, recipe = s4.package, s4.recipe
        else:
            package, recipe = Path(args.package), None
        files_sha = published.assemble(package, published.tokenizer_source(), MODEL_DIR)
        expected_revision = "file-" + files_sha[:16]  # EmbeddingGemma.modelRevision for a .file source
        log(f"bundling {package} as {MODEL_DIR} (files.json {files_sha[:16]})")
        ctl = Devicectl()
        device = pick_device(ctl.list_devices(), args.device)
        result = benchmark(device, build_app(device, out), out, ctl)
        if result.get("model_revision") != expected_revision:
            raise RuntimeError(f"the app ran model {result.get('model_revision')!r}, expected {expected_revision!r}")
        (out / "devices").mkdir(exist_ok=True)
        (out / "devices" / f"{device.product_type}.json").write_text(json.dumps(result, indent=1) + "\n")
        metrics, details = summarize([result], targets(STAGE)["target_ms_per_image_70"])
        report.metrics.update(metrics)
        report.details.update(details, package=str(package), s4_recipe=recipe, files_json_sha256=files_sha,
                              bundle_id=BUNDLE_ID)
    return fill


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", help="UDID or CoreDevice identifier (default: the first physical iPhone)")
    p.add_argument("--package", help="the .mlpackage to bundle (default: the one s4's report says it shipped)")
    run_stage(STAGE, make_fill(p.parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
