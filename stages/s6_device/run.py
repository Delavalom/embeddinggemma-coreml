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
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from egcm import manifest, published
from egcm.buckets import BUDGETS
from egcm.gates import targets
from egcm.paths import ROOT, stage_dir
from egcm.report import Report
from egcm.stage import run_stage
from stages.s6_device.device import (
    BUNDLE_ID,
    COMPILE_RESULT_FILE,
    RESULT_FILE,
    TIMEOUT_SECONDS,
    Device,
    Devicectl,
    pick_device,
    poll_result,
)
from stages.s6_device.team import ensure_team_xcconfig

STAGE = "s6_device"
DEMO = ROOT / "Examples" / "PhotoSearchDemo"
# The published folders (egcm.published) the app bundles as its `Model` resource: Model/b70/ and Model/b280/
# (ADR-0010). Gitignored; written by this stage.
MODEL_DIR = DEMO / "Model"
# A function load in the benchmark slower than this was a compile, not a warm load (ADR-0009). Measured on the
# iPhone 13 Pro Max: cold compiles 45-240 s per function, warm loads of the whole 14-function package 6.5-7 s.
MAX_WARM_LOAD_MS = 30_000
# The compile run: 18 cold compiles (4 text + 5 image per budget package), 45-240 s each on the iPhone 13 Pro Max,
# plus the cooldown before a retry (FunctionLoad.cooldown). The benchmark keeps device.TIMEOUT_SECONDS.
COMPILE_TIMEOUT_SECONDS = 120 * 60
# A failed compile run is relaunched (a new process; functions already compiled stay cached and load warm), up to
# this many launches, after a pause for the phone to cool. On iOS 27.0.1 a hot phone failed one Neural Engine compile
# three times in a row within one process, even with FunctionLoad's cooldown; Roll compiles across launches the same
# way. The cold-compile gate takes the highest peak over the launches.
COMPILE_LAUNCHES = 3
COMPILE_RELAUNCH_PAUSE_SECONDS = 180


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
    for rel in bundled_layouts():
        if not (app / "Model" / rel / published.FILES_JSON).is_file():
            raise RuntimeError(f"{app} has no {Path('Model', rel, published.FILES_JSON)}; the bundling script did "
                               "not run")
    _check(["codesign", "--verify", "--deep", "--strict", str(app)], DEMO, out / "codesign.log")
    return app


class PhoneLocked(RuntimeError):
    """devicectl could not start the app: the phone is locked (iOS answers "failed to launch ... Invalid argument")."""


# How long `launch` waits for the phone to be unlocked, retrying every UNLOCK_POLL_SECONDS.
UNLOCK_WAIT_SECONDS = 60 * 60
UNLOCK_POLL_SECONDS = 30


def _launch_once(device: Device, out: Path, ctl: Devicectl, mode: str, source: str, timeout: float) -> dict:
    run_id = uuid.uuid4().hex
    console = out / f"console_{device.product_type}{mode}.log"
    with open(console, "w") as f:
        proc = subprocess.Popen(ctl.launch_command(device, run_id, mode), stdout=f, stderr=subprocess.STDOUT)
        log(f"launched {mode}. Keep the phone unlocked. Console: {console}")
        try:
            return poll_result(lambda: ctl.fetch_result(device, out / Path(source).name, source), run_id,
                               app_exited=proc.poll, log=log, name=source, timeout=timeout)
        except RuntimeError as e:
            if "failed to launch" in console.read_text(errors="replace"):
                raise PhoneLocked(f"could not launch {mode}: unlock the phone") from e
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()


def launch(device: Device, out: Path, ctl: Devicectl, mode: str, source: str,
           timeout: float = TIMEOUT_SECONDS, sleep: Callable[[float], None] = time.sleep) -> dict:
    """Launches the installed app with `mode` (`-compile` or `-benchmark`) and waits up to `timeout` s for this
    run's `source`. While the phone is locked it retries every UNLOCK_POLL_SECONDS, up to UNLOCK_WAIT_SECONDS."""
    waited = 0
    while True:
        try:
            return _launch_once(device, out, ctl, mode, source, timeout)
        except PhoneLocked:
            if waited >= UNLOCK_WAIT_SECONDS:
                raise
            if waited == 0:
                log("the phone is locked: unlock it (keep it on the cable); retrying every "
                    f"{UNLOCK_POLL_SECONDS} s")
            sleep(UNLOCK_POLL_SECONDS)
            waited += UNLOCK_POLL_SECONDS


def benchmark(device: Device, app: Path, out: Path, ctl: Devicectl | None = None) -> dict:
    """Installs, compiles every function in one launch (`-compile`), then benchmarks in a new process
    (`-benchmark`), so the benchmark's loads are warm (ADR-0009). Returns the benchmark's result with the compile
    run's under `compile`."""
    ctl = ctl or Devicectl()
    log(f"installing on {device.marketing_name} ({device.product_type}, iOS {device.os_version})")
    ctl.install(device, app)
    log("compiling every function (minutes on a fresh install)")
    compiled = compile_all(device, out, ctl)
    log("benchmarking. Tap Allow Full Access on the phone if it asks for Photos (first run only)")
    result = launch(device, out, ctl, "-benchmark", RESULT_FILE)
    return result | {"compile": compiled}


def compile_all(device: Device, out: Path, ctl: Devicectl, launches: int = COMPILE_LAUNCHES,
                pause: Callable[[float], None] = time.sleep) -> dict:
    """The `-compile` run, relaunched after a failure up to `launches` times. Returns the successful run's result
    with `peak_rss_mb` the highest over every launch, `function_ms` the first (cold) time per function, `launches`
    and `launch_errors`. A timeout is not retried."""
    tries: list[dict] = []
    partial = out / Path(COMPILE_RESULT_FILE).name
    for n in range(1, launches + 1):
        try:
            r = launch(device, out, ctl, "-compile", COMPILE_RESULT_FILE, COMPILE_TIMEOUT_SECONDS)
        except RuntimeError as e:
            failed = json.loads(partial.read_text()) if partial.is_file() else {}
            tries.append(failed | {"error": str(e)})
            if n == launches:
                raise RuntimeError(f"the compile run failed in {launches} launches; last: {e}") from e
            log(f"compile launch {n} failed ({e}); relaunching in {COMPILE_RELAUNCH_PAUSE_SECONDS} s")
            pause(COMPILE_RELAUNCH_PAUSE_SECONDS)
            continue
        tries.append(r)
        return merge_compile_runs(tries)
    raise AssertionError("unreachable")


def merge_compile_runs(tries: list[dict]) -> dict:
    """The last (successful) run, with the peak over all and each function's first recorded time."""
    first: dict[str, float] = {}
    for t in tries:
        for name, ms in t.get("function_ms", {}).items():
            first.setdefault(name, ms)
    peaks = [p for t in tries if (p := t.get("peak_rss_mb")) is not None]
    return tries[-1] | {"peak_rss_mb": max(peaks) if peaks else None, "function_ms": first, "launches": len(tries),
                        "launch_errors": [t["error"] for t in tries if "error" in t]}


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


def bundled_layouts() -> list[str]:
    """The published layouts under MODEL_DIR, relative to it: `b70`, `b280` (ADR-0010), or `.` (one flat layout)."""
    return sorted(p.parent.relative_to(MODEL_DIR).as_posix() for p in MODEL_DIR.glob("**/" + published.FILES_JSON))


def bundle(args: argparse.Namespace, out: Path) -> tuple[dict[str, str], dict]:
    """Writes MODEL_DIR (replacing it) and returns (budget -> the `modelRevision` the app must report, details).

    Default: one layout per budget, `Model/b70/` and `Model/b280/`, from s4's device archives, unpacked
    (sha256-checked): the bytes each phone downloads. The app's `.file(Model)` source picks `Model/b<budget>/`, so
    each budget reports `file-<sha256 prefix of its own files.json>`. `--package`: that one package (all 14
    functions), flat in `Model/`, for both budgets."""
    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    if args.package is not None:
        sha = published.assemble(Path(args.package), published.tokenizer_source(), MODEL_DIR)
        log(f"bundling {args.package} as {MODEL_DIR} (files.json {sha[:16]})")
        return ({str(b): "file-" + sha[:16] for b in BUDGETS},
                {"package": str(args.package), "s4_recipe": None, "files_json_sha256": {"all": sha}})
    s4 = published.shipped()
    shas, packages = {}, {}
    for b, v in sorted(s4.budgets.items()):
        unpacked = manifest.unpack(v.archive, out / "unpacked" / published.budget_dir(b), v.archive_sha256)
        tokenizer = published.check_tokenizer(unpacked / published.TOKENIZER_DIR)
        packages[str(b)] = str(unpacked / published.PACKAGE_NAME)
        shas[str(b)] = published.assemble(unpacked / published.PACKAGE_NAME, tokenizer,
                                          MODEL_DIR / published.budget_dir(b))
        log(f"bundling {v.archive.name} as {MODEL_DIR / published.budget_dir(b)} (files.json {shas[str(b)][:16]})")
    return ({b: "file-" + sha[:16] for b, sha in shas.items()},
            {"package": packages, "s4_recipe": s4.recipe, "files_json_sha256": shas})


def check_revisions(result: dict, expected: dict[str, str]) -> None:
    """Both launches must report, per budget, the model this run bundled (`model_revisions`: budget -> revision)."""
    for run, revisions in (("benchmark", result.get("model_revisions")),
                           ("compile", result.get("compile", {}).get("model_revisions"))):
        if revisions != expected:
            raise RuntimeError(f"the {run} ran models {revisions!r}, expected {expected!r}")


def make_fill(args: argparse.Namespace):
    def fill(report: Report) -> None:
        out = stage_dir(STAGE)
        expected, bundled = bundle(args, out)
        ctl = Devicectl()
        device = pick_device(ctl.list_devices(), args.device)
        result = benchmark(device, build_app(device, out), out, ctl)
        check_revisions(result, expected)
        if cold := cold_loads(result):
            raise RuntimeError(f"the benchmark compiled {', '.join(cold)} (over {MAX_WARM_LOAD_MS} ms) after the "
                               "compile run; its peak is not a warm peak")
        (out / "devices").mkdir(exist_ok=True)
        (out / "devices" / f"{device.product_type}.json").write_text(json.dumps(result, indent=1) + "\n")
        metrics, details = summarize([result], targets(STAGE)["target_ms_per_image_70"])
        report.metrics.update(metrics)
        report.details.update(details, **bundled, expected_revisions=expected, bundle_id=BUNDLE_ID)
    return fill


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", help="UDID or CoreDevice identifier (default: the first physical iPhone)")
    p.add_argument("--package", help="one 14-function .mlpackage to bundle flat in Model/ for both budgets (default: "
                                     "s4's two budget packages, Model/b70/ and Model/b280/)")
    run_stage(STAGE, make_fill(p.parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
