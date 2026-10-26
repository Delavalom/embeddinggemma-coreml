# group: convert
"""s6_device: PhotoSearchDemo's benchmark on a real iPhone.

Gates (gates.toml [s6_device], ADR-0009): max_peak_rss_mb (the warm benchmark's peak, maximum over devices),
max_cold_compile_peak_mb (the compile run's peak after a fresh install), min_devices (devices with a result).
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

from egcm import manifest, published
from egcm.gates import targets
from egcm.paths import ROOT, stage_dir
from egcm.report import Report
from egcm.stage import run_stage
from stages.s6_device.device import (
    BUNDLE_ID,
    COMPILE_RESULT_FILE,
    RESULT_FILE,
    Device,
    Devicectl,
    pick_device,
    poll_result,
)
from stages.s6_device.team import ensure_team_xcconfig

STAGE = "s6_device"
DEMO = ROOT / "Examples" / "PhotoSearchDemo"
# The published folder (egcm.published) the app bundles as its `Model` resource. Gitignored; written by this stage.
MODEL_DIR = DEMO / "Model"
# A function load in the benchmark slower than this was a compile, not a warm load (ADR-0009). Measured on the
# iPhone 13 Pro Max: cold compiles 45-240 s per function, warm loads of the whole 14-function package 6.5-7 s.
MAX_WARM_LOAD_MS = 30_000


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


def launch(device: Device, out: Path, ctl: Devicectl, mode: str, source: str) -> dict:
    """Launches the installed app with `mode` (`-compile` or `-benchmark`) and waits for this run's `source`."""
    run_id = uuid.uuid4().hex
    console = out / f"console_{device.product_type}{mode}.log"
    with open(console, "w") as f:
        proc = subprocess.Popen(ctl.launch_command(device, run_id, mode), stdout=f, stderr=subprocess.STDOUT)
        log(f"launched {mode}. Keep the phone unlocked. Console: {console}")
        try:
            return poll_result(lambda: ctl.fetch_result(device, out / Path(source).name, source), run_id,
                               app_exited=proc.poll, log=log, name=source)
        finally:
            if proc.poll() is None:
                proc.terminate()


def benchmark(device: Device, app: Path, out: Path, ctl: Devicectl | None = None) -> dict:
    """Installs, compiles every function in one launch (`-compile`), then benchmarks in a new process
    (`-benchmark`), so the benchmark's loads are warm (ADR-0009). Returns the benchmark's result with the compile
    run's under `compile`."""
    ctl = ctl or Devicectl()
    log(f"installing on {device.marketing_name} ({device.product_type}, iOS {device.os_version})")
    ctl.install(device, app)
    log("compiling every function (minutes on a fresh install)")
    compiled = launch(device, out, ctl, "-compile", COMPILE_RESULT_FILE)
    log("benchmarking. Tap Allow Full Access on the phone if it asks for Photos (first run only)")
    result = launch(device, out, ctl, "-benchmark", RESULT_FILE)
    return result | {"compile": compiled}


def cold_loads(result: dict, limit_ms: float = MAX_WARM_LOAD_MS) -> list[str]:
    """`budget/function` for every benchmark load slower than `limit_ms`: compiles the compile run did not cache."""
    return [f"{b}/{name}" for b, phase in sorted(result.get("budgets", {}).items())
            for name, ms in sorted(phase.get("function_ms", {}).items()) if ms > limit_ms]


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
        "max_cold_compile_peak_mb": _worst(results, lambda r: r.get("compile", {}).get("peak_rss_mb")),
        "ms_per_image_70_median": _worst(results, ms("70")),
        "ms_per_image_280_median": _worst(results, ms("280")),
        "ms_per_query_median": _worst(results, lambda r: r.get("query", {}).get("ms", {}).get("median")),
    }
    details = {
        "target_ms_per_image_70": target_ms_70,
        "meets_target_70": {r["device_model"]: (m := ms("70")(r)) is not None and m <= target_ms_70 for r in results},
        "available_memory_mb_at_launch": {r["device_model"]: r.get("compile", {}).get("available_memory_mb_at_launch")
                                          for r in results},
        "devices": {r["device_model"]: r for r in results},
    }
    return metrics, details


def make_fill(args: argparse.Namespace):
    def fill(report: Report) -> None:
        out = stage_dir(STAGE)
        if args.package is None:  # s4's device archive, unpacked (sha256-checked): the bytes a phone downloads
            s4 = published.shipped()
            unpacked = manifest.unpack(s4.archive, out / "unpacked", s4.archive_sha256)
            package, recipe = unpacked / published.PACKAGE_NAME, s4.recipe
            tokenizer = published.check_tokenizer(unpacked / published.TOKENIZER_DIR)
        else:
            package, recipe, tokenizer = Path(args.package), None, published.tokenizer_source()
        files_sha = published.assemble(package, tokenizer, MODEL_DIR)
        expected_revision = "file-" + files_sha[:16]  # EmbeddingGemma.modelRevision for a .file source
        log(f"bundling {package} as {MODEL_DIR} (files.json {files_sha[:16]})")
        ctl = Devicectl()
        device = pick_device(ctl.list_devices(), args.device)
        result = benchmark(device, build_app(device, out), out, ctl)
        for run, revision in (("benchmark", result.get("model_revision")),
                              ("compile", result.get("compile", {}).get("model_revision"))):
            if revision != expected_revision:
                raise RuntimeError(f"the {run} ran model {revision!r}, expected {expected_revision!r}")
        if cold := cold_loads(result):
            raise RuntimeError(f"the benchmark compiled {', '.join(cold)} (over {MAX_WARM_LOAD_MS} ms) after the "
                               "compile run; its peak is not a warm peak")
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
