"""s6 without a device: devicectl parsing, polling, the summary that feeds the gates, and the published folder."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from egcm import published
from egcm.gates import evaluate, targets
from egcm.report import Report
from stages.s6_device import device as dev
from stages.s6_device import run as s6
from stages.s6_device import team

DATA = Path(__file__).parent / "data"


def listing():
    return json.loads((DATA / "devicectl_list_devices.json").read_text())


def test_parse_devices_keeps_physical_with_developer_mode():
    devices = dev.parse_devices(listing())
    assert devices == [dev.Device(identifier="AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE", udid="00008110-000000000000001E",
                                  product_type="iPhone14,3", marketing_name="iPhone 13 Pro Max", os_version="27.0.1")]


def test_parse_devices_skips_developer_mode_off_and_non_ios():
    data = listing()
    phone = data["result"]["devices"][1]
    off, mac = copy.deepcopy(phone), copy.deepcopy(phone)
    off["deviceProperties"]["developerModeStatus"] = "disabled"
    mac["hardwareProperties"]["platform"] = "macOS"
    data["result"]["devices"] = [off, mac]
    assert dev.parse_devices(data) == []
    assert dev.parse_devices({}) == []


def test_pick_device():
    devices = dev.parse_devices(listing())
    assert dev.pick_device(devices) == devices[0]
    assert dev.pick_device(devices, "00008110-000000000000001E") == devices[0]
    assert dev.pick_device(devices, devices[0].identifier) == devices[0]
    with pytest.raises(RuntimeError, match="Developer Mode"):
        dev.pick_device([])
    with pytest.raises(RuntimeError, match="nope"):
        dev.pick_device(devices, "nope")


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def poll(responses, clock, **kw):
    it = iter(responses)
    return dev.poll_result(lambda: next(it, None), "run1", sleep=clock.sleep, clock=clock, log=lambda _: None, **kw)


def test_poll_ignores_missing_and_old_results_then_returns():
    clock = FakeClock()
    done = {"run_id": "run1", "status": "done", "peak_rss_mb": 1.0}
    got = poll([None, {"run_id": "old", "status": "done"}, {"run_id": "run1", "status": "running"}, done], clock)
    assert got == done
    assert clock.sleeps == [15, 15, 15]


def test_poll_times_out_after_30_minutes():
    clock = FakeClock()
    with pytest.raises(TimeoutError, match="30 min"):
        poll([], clock)
    assert len(clock.sleeps) == 30 * 60 // 15


def test_poll_raises_the_apps_error():
    with pytest.raises(RuntimeError, match="Photos access was denied"):
        poll([{"run_id": "run1", "status": "error", "error": "Photos access was denied"}], FakeClock())


def test_poll_stops_when_the_app_exits_without_a_result():
    clock = FakeClock()
    with pytest.raises(RuntimeError, match="exited"):
        poll([None, None], clock, app_exited=lambda: 0 if clock.t >= 15 else None)
    assert clock.sleeps == [15]


def test_devicectl_commands_and_fetch(tmp_path):
    calls = []
    d = dev.parse_devices(listing())[0]

    def run(cmd):
        calls.append(cmd)
        if "copy" in cmd:
            Path(cmd[cmd.index("--destination") + 1]).write_text('{"run_id": "r"}')
        return subprocess.CompletedProcess(cmd, 0, "", "")

    ctl = dev.Devicectl(run)
    assert ctl.fetch_result(d, tmp_path / "r.json") == {"run_id": "r"}
    copy_cmd = calls[-1]
    assert copy_cmd[copy_cmd.index("--domain-identifier") + 1] == dev.BUNDLE_ID
    assert copy_cmd[copy_cmd.index("--source") + 1] == "Documents/s6_device.json"
    launch = ctl.launch_command(d, "abc")
    assert "--console" in launch and launch[-4:] == ["--", "-benchmark", "-runID", "abc"]
    ctl.install(d, tmp_path / "A.app")
    assert calls[-1][:5] == ["xcrun", "devicectl", "device", "install", "app"]


def test_fetch_returns_none_when_copy_fails(tmp_path):
    ctl = dev.Devicectl(lambda cmd: subprocess.CompletedProcess(cmd, 1, "", "no such file"))
    assert ctl.fetch_result(dev.parse_devices(listing())[0], tmp_path / "r.json") is None


def result(model="iPhone14,3", rss=512.0, ms70=48.0):
    return {"run_id": "r", "status": "done", "device_model": model, "peak_rss_mb": rss,
            "budgets": {"70": {"images": 1000, "ms_per_image": {"median": ms70, "p90": ms70 * 1.2}},
                        "280": {"images": 200, "ms_per_image": {"median": 140.0, "p90": 160.0}}},
            "query": {"queries": 20, "ms": {"median": 9.0, "p90": 12.0}}}


def test_summarize_and_gates():
    target = targets("s6_device")["target_ms_per_image_70"]
    metrics, details = s6.summarize([result(), result("iPhone16,1", rss=700.0, ms70=80.0)], target)
    assert metrics["min_devices"] == 2
    assert metrics["max_peak_rss_mb"] == 700.0
    assert metrics["ms_per_image_70_median"] == 80.0
    assert metrics["ms_per_query_median"] == 9.0
    assert details["target_ms_per_image_70"] == target
    assert details["meets_target_70"] == {"iPhone14,3": True, "iPhone16,1": False}
    r = Report(stage="s6_device", metrics=metrics)
    assert evaluate(r, "s6_device").passed  # speed is recorded, not gated

    heavy, _ = s6.summarize([result(rss=900.0)], target)
    r = evaluate(Report(stage="s6_device", metrics=heavy), "s6_device")
    assert not r.passed and r.failures == ["max_peak_rss_mb: 900.0 not <= 800"]

    none, _ = s6.summarize([], target)
    r = evaluate(Report(stage="s6_device", metrics=none), "s6_device")
    assert not r.passed and "max_peak_rss_mb: metric missing" in r.failures


def test_fill_rejects_another_models_result(tmp_path, monkeypatch):
    pkg = tmp_path / "EmbeddingGemma2.mlpackage"
    (pkg / "Data").mkdir(parents=True)
    (pkg / "Manifest.json").write_text("{}")
    tok = tmp_path / "tok"
    tok.mkdir()
    for name in published.TOKENIZER_FILES:
        (tok / name).write_text(name)
    phone = dev.parse_devices(listing())[0]
    monkeypatch.setattr(s6, "MODEL_DIR", tmp_path / "Model")
    monkeypatch.setattr(s6, "stage_dir", lambda _: tmp_path / "out")
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(s6.published, "tokenizer_source", lambda: tok)
    monkeypatch.setattr(s6.Devicectl, "list_devices", lambda self: [phone])
    monkeypatch.setattr(s6, "build_app", lambda d, out: tmp_path / "A.app")
    monkeypatch.setattr(s6, "benchmark", lambda d, app, out, ctl: result() | {"model_revision": "file-other"})
    fill = s6.make_fill(s6.argparse.Namespace(package=str(pkg), device=None))
    with pytest.raises(RuntimeError, match="expected 'file-"):
        fill(Report(stage="s6_device"))

    entries = json.loads((tmp_path / "Model" / "files.json").read_text())
    expected = published.sha256_file(tmp_path / "Model" / "files.json")
    ours = result() | {"model_revision": "file-" + expected[:16]}
    monkeypatch.setattr(s6, "benchmark", lambda d, app, out, ctl: ours)
    r = Report(stage="s6_device")
    fill(r)
    assert r.metrics["min_devices"] == 1 and r.details["files_json_sha256"] == expected
    assert (tmp_path / "out" / "devices" / "iPhone14,3.json").is_file()
    assert [e["path"] for e in entries] == ["EmbeddingGemma2.mlpackage/Manifest.json", "tokenizer/tokenizer.json",
                                        "tokenizer/tokenizer_config.json"]


def test_files_manifest_skips_hidden_and_itself(tmp_path):
    (tmp_path / "a" / ".git").mkdir(parents=True)
    (tmp_path / "a" / ".git" / "x").write_text("x")
    (tmp_path / "a" / ".DS_Store").write_text("x")
    (tmp_path / "a" / "b.bin").write_bytes(b"hello")
    (tmp_path / "files.json").write_text("[]")
    entries = published.files_manifest(tmp_path)
    assert entries == [{"path": "a/b.bin", "size": 5,
                        "sha256": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"}]


def test_tokenizer_source_verifies_hashes(tmp_path):
    d = tmp_path / "artifacts" / "tokenizer" / published.HF_REVISION
    d.mkdir(parents=True)
    for name in published.TOKENIZER_FILES:
        (d / name).write_text("tampered")
    with pytest.raises(RuntimeError, match="do not match"):
        published.tokenizer_source(tmp_path)
    with pytest.raises(FileNotFoundError, match="just stage 0"):
        published.tokenizer_source(tmp_path / "empty")


def test_team_ids_from_certificate_subjects():
    certs = {"data": [{"attributes": {"certificateContent": "AAAA"}}, {"attributes": {"certificateContent": "AAAB"}},
                      {"attributes": {}}]}
    subject = "subject=\n    userId                    = X\n    organizationalUnitName    = ABCDE12345\n"
    assert team.team_ids(certs, subject_of=lambda der: subject) == {"ABCDE12345"}
    assert "DEVELOPMENT_TEAM = ABCDE12345" in team.render("ABCDE12345")


def test_build_command_is_release_for_the_device():
    cmd = s6.build_command("UDID1", Path("/tmp/dd"))
    assert cmd[cmd.index("-destination") + 1] == "id=UDID1"
    assert cmd[cmd.index("-configuration") + 1] == "Release" and "-allowProvisioningUpdates" in cmd
