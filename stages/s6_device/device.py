"""The iPhone side of s6: find the device, install and launch the demo, fetch its result file.

Everything goes through `xcrun devicectl` (Xcode 15+). The pure parts (parsing `devicectl list devices` JSON,
polling for the result) take their I/O as arguments so tests run without a device.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

BUNDLE_ID = "com.delavalom.egcm.demo"
RESULT_FILE = "Documents/s6_device.json"
# The `-compile` launch's result (ADR-0009).
COMPILE_RESULT_FILE = "Documents/s6_compile.json"
POLL_SECONDS = 15
TIMEOUT_SECONDS = 30 * 60
NO_DEVICE = ("connect an iPhone with Developer Mode (Settings > Privacy & Security > Developer Mode), "
             "unlocked and trusted")


@dataclass(frozen=True)
class Device:
    identifier: str  # CoreDevice identifier, for `devicectl --device`
    udid: str  # hardware UDID, for `xcodebuild -destination id=`
    product_type: str  # e.g. "iPhone14,3"
    marketing_name: str
    os_version: str


def parse_devices(listing: dict) -> list[Device]:
    """Physical iOS devices with Developer Mode on, in `devicectl list devices --json-output` order."""
    out = []
    for d in listing.get("result", {}).get("devices", []):
        hw, props = d.get("hardwareProperties", {}), d.get("deviceProperties", {})
        if hw.get("reality") != "physical" or hw.get("platform") != "iOS":
            continue
        if props.get("developerModeStatus") != "enabled":
            continue
        out.append(Device(identifier=d["identifier"], udid=hw.get("udid", d["identifier"]),
                          product_type=hw.get("productType", "unknown"),
                          marketing_name=hw.get("marketingName", "unknown"),
                          os_version=props.get("osVersionNumber", "unknown")))
    return out


def pick_device(devices: list[Device], want: str | None = None) -> Device:
    """The first device, or the one whose identifier or UDID is `want`."""
    if want:
        for d in devices:
            if want in (d.identifier, d.udid):
                return d
        raise RuntimeError(f"no physical iOS device {want!r} with Developer Mode; {NO_DEVICE}")
    if not devices:
        raise RuntimeError(NO_DEVICE)
    return devices[0]


def poll_result(fetch: Callable[[], dict | None], run_id: str, *, interval: float = POLL_SECONDS,
                timeout: float = TIMEOUT_SECONDS, app_exited: Callable[[], int | None] = lambda: None,
                sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                log: Callable[[str], None] = print, name: str = RESULT_FILE) -> dict:
    """Calls `fetch` every `interval` s until it returns this run's result (`run_id` matches), up to `timeout` s.

    A result from an older run (other `run_id`) is ignored. `status == "error"` raises with the app's message.
    If `app_exited()` returns an exit code, one last fetch is made and then it raises."""
    deadline = clock() + timeout
    while True:
        exited = app_exited()
        r = fetch()
        if r is not None and r.get("run_id") == run_id:
            if r.get("status") == "error":
                raise RuntimeError(f"the app failed on the device ({name}): {r.get('error', 'no message')}")
            if r.get("status") == "done":
                return r
        if exited is not None:
            raise RuntimeError(f"the app exited (code {exited}) without writing a result for run {run_id}")
        if clock() >= deadline:
            raise TimeoutError(f"no result for run {run_id} after {timeout / 60:.0f} min")
        log(f"waiting for {name} on the device ({(deadline - clock()) / 60:.0f} min left)")
        sleep(interval)


Runner = Callable[[list[str]], subprocess.CompletedProcess]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


class Devicectl:
    """Thin wrapper over `xcrun devicectl`. `run` is injectable for tests."""

    def __init__(self, run: Runner = _run):
        self.run = run

    def _check(self, cmd: list[str]) -> subprocess.CompletedProcess:
        r = self.run(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd[:4])} failed ({r.returncode}): {(r.stderr or r.stdout).strip()}")
        return r

    def list_devices(self) -> list[Device]:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "devices.json"
            self._check(["xcrun", "devicectl", "list", "devices", "--json-output", str(out)])
            return parse_devices(json.loads(out.read_text()))

    def install(self, device: Device, app: Path) -> None:
        self._check(["xcrun", "devicectl", "device", "install", "app", "--device", device.identifier, str(app)])

    def launch_command(self, device: Device, run_id: str, mode: str = "-benchmark") -> list[str]:
        """Foreground launch with the app's console attached (it returns when the app exits). `mode` is
        `-benchmark` or `-compile`. `--` keeps devicectl from reading the app's arguments as its own options; the
        app ignores a literal `--`."""
        return ["xcrun", "devicectl", "device", "process", "launch", "--console", "--terminate-existing",
                "--device", device.identifier, BUNDLE_ID, "--", mode, "-runID", run_id]

    def fetch_result(self, device: Device, dest: Path, source: str = RESULT_FILE) -> dict | None:
        """Copies `source` (default Documents/s6_device.json) from the app's container; None if it is not there
        (yet) or unreadable."""
        dest.unlink(missing_ok=True)
        r = self.run(["xcrun", "devicectl", "device", "copy", "from", "--device", device.identifier,
                      "--domain-type", "appDataContainer", "--domain-identifier", BUNDLE_ID,
                      "--source", source, "--destination", str(dest)])
        if r.returncode != 0 or not dest.is_file():
            return None
        try:
            return json.loads(dest.read_text())
        except json.JSONDecodeError:
            return None
