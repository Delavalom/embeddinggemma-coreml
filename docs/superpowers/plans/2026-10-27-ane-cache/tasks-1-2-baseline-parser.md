<!-- Part of docs/superpowers/plans/2026-10-27-ios27-ane-cache-investigation.md; read that first. -->

## Task 1: Phone baseline and reset

**Files:**
- Create: `docs/investigations/evidence/2026-10-27-ane-cache/phone-baseline-<HHMM>.txt`

- [ ] **Step 1: Record the phone's state before changing anything.**

```bash
U=${UDID:?set UDID}; D=~/egcm-evidence-raw/$(date +%F); mkdir -p $D
xcrun devicectl device info details --device $U --json-output $D/details-$(date +%H%M).json >/dev/null
xcrun devicectl device info files --device $U --domain-type appDataContainer \
  --domain-identifier com.delavalom.egcm.demo --subdirectory Library --json-output $D/listing-pre-uninstall.json >/dev/null
```

From the details JSON, record the OS build (expect 24A446) and free capacity. If the JSON has no free-space field,
note "not in devicectl" and use the probe's `volumeAvailableCapacityForImportantUsage` from Task 7 instead.

- [ ] **Step 2: Uninstall the demo/probe app and reboot the phone.**

```bash
xcrun devicectl device uninstall app --device $U com.delavalom.egcm.demo
```

Ask the founder to restart the phone and to free space until Settings → General → iPhone Storage shows ≥ 8 GB
available. Record both numbers (before and after the uninstall) in `phone-baseline-<HHMM>.txt`. This also starts on
handoff open question U4 (F9: only ~1.9 GB was free after the last uninstall).

- [ ] **Step 3: Commit.**

```bash
git add docs/investigations/evidence/2026-10-27-ane-cache/phone-baseline-*.txt
git commit -m "evidence(ane-cache): phone baseline before the experiments"
```

## Task 2: Cache-listing parser (`e5_listing.py`)

One parser for both the phone listing (devicectl JSON) and the Mac folder, so every experiment reports the same
fields.

**Files:**
- Create: `scripts/probes/e5_listing.py`
- Create: `scripts/probes/__init__.py` (empty)
- Test: `tests/test_e5_listing.py`

- [ ] **Step 1: Write the failing test.**

```python
# tests/test_e5_listing.py
import json

from scripts.probes.e5_listing import diff, from_devicectl, summarize

K1 = "A" * 64
K2 = "B" * 64
B1 = "C" * 64
PFX = "Caches/com.delavalom.egcm.demo/com.apple.e5rt.e5bundlecache/24A446/"


def entry(rel, size, is_dir=False, mtime="2026-10-27T15:00:00.000Z"):
    return {"relativePath": PFX + rel, "metadata": {"size": size, "lastModDate": mtime},
            "resources": {"isDirectory": is_dir}}


def listing(entries):
    return {"result": {"files": entries}}


def test_milhash_bundle_with_payload_and_skeleton(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(json.dumps(listing([
        entry(K1, 96, True), entry(f"{K1}/model.milhash", 64),
        entry(K2, 96, True), entry(f"{K2}/{B1}.bundle", 96, True),
        entry(f"{K2}/{B1}.bundle/H14.bundle", 96, True),
        entry(f"{K2}/{B1}.bundle/H14.bundle/H14.e5", 5192),
        entry(f"{K2}/{B1}.bundle/H14.bundle/embed_text_16", 128, True),
        entry(f"{K2}/{B1}.bundle/H14.bundle/embed_text_16/embed_text_16_bnns", 96, True),
        entry(f"{K2}/{B1}.bundle/H14.bundle/embed_text_16/embed_text_16_bnns/bnns_program.bnnsir", 268468224),
    ])))
    s = summarize(from_devicectl(p))
    assert s["keys"][K1]["kind"] == "milhash"
    b = s["keys"][K2]["bundles"][0]
    assert b["functions"] == ["embed_text_16"]
    assert b["has_e5"] is True and b["payload_bytes"] == 268468224 + 5192 and b["tmp"] is False
    assert s["totals"] == {"keys": 2, "milhash": 1, "bundles": 1, "bundles_with_payload": 1, "tmp_bundles": 0}


def test_skeleton_bundle_has_no_payload(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(json.dumps(listing([
        entry(K2, 96, True), entry(f"{K2}/{B1}.bundle", 96, True),
        entry(f"{K2}/{B1}.bundle/H14.bundle/embed_text_16/embed_text_16_ane", 64, True),
    ])))
    b = summarize(from_devicectl(p))["keys"][K2]["bundles"][0]
    assert b["has_e5"] is False and b["payload_bytes"] == 0


def test_diff_reports_new_keys_and_lost_payload(tmp_path):
    before = {"keys": {K2: {"kind": "bundle", "bundles": [{"hash": B1, "payload_bytes": 10, "functions": ["f"]}]}}}
    after = {"keys": {K2: {"kind": "bundle", "bundles": [{"hash": B1, "payload_bytes": 0, "functions": ["f"]}]},
                      K1: {"kind": "milhash", "bundles": []}}}
    d = diff(before, after)
    assert d["added_keys"] == [K1] and d["removed_keys"] == []
    assert d["payload_lost"] == [{"key": K2, "bundle": B1, "functions": ["f"], "bytes_before": 10}]
```

- [ ] **Step 2: Run it and check that it fails.**

Run: `uv run --exact --group convert pytest tests/test_e5_listing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.probes'`. Then create an empty
`scripts/probes/__init__.py` (`scripts/__init__.py` exists; the repo runs scripts as modules, AGENTS.md).

- [ ] **Step 3: Write the implementation.**

```python
# scripts/probes/e5_listing.py
"""Summarise a Core ML E5 bundle cache (com.apple.e5rt.e5bundlecache) from a phone or a Mac.

Phone: `xcrun devicectl device info files ... --subdirectory Library --json-output x.json`, then `summarize x.json`.
Mac:   `summarize ~/Library/Caches/<executable>/com.apple.e5rt.e5bundlecache`.
Layout seen on 2026-10-27: <build>/<key>/model.milhash, or <build>/<key>/<hash>[.tmp.<pid>_<n>].bundle/<arch>.bundle/
{<arch>.e5, <fn>/<fn>_ane/..., <fn>/<fn>_bnns/bnns_program.bnnsir}. "Payload" = every regular file under a bundle.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

CACHE = "com.apple.e5rt.e5bundlecache/"
KEY = re.compile(r"^[0-9A-F]{64}$")


@dataclass(frozen=True)
class Entry:
    rel: str  # path below <cache>/, e.g. "24A446/<key>/..."
    size: int
    mtime: str
    is_dir: bool


def from_devicectl(path: Path) -> list[Entry]:
    out = []
    for f in json.loads(Path(path).read_text())["result"]["files"]:
        rp = f["relativePath"]
        if CACHE in rp:
            out.append(Entry(rp.split(CACHE, 1)[1], f["metadata"]["size"], f["metadata"]["lastModDate"],
                             f["resources"]["isDirectory"]))
    return out


def from_dir(root: Path) -> list[Entry]:
    from datetime import datetime, timezone
    out = []
    for p in sorted(Path(root).rglob("*")):
        st = p.lstat()
        out.append(Entry(str(p.relative_to(root)), st.st_size,
                         datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(), p.is_dir()))
    return out


def summarize(entries: list[Entry]) -> dict:
    keys: dict[str, dict] = {}
    for e in entries:
        parts = e.rel.split("/")
        if len(parts) < 2 or not KEY.match(parts[1]):
            continue
        k = keys.setdefault(parts[1], {"kind": "empty", "bundles": [], "newest": e.mtime, "_b": {}})
        k["newest"] = max(k["newest"], e.mtime)
        if len(parts) == 3 and parts[2] == "model.milhash":
            k["kind"] = "milhash"
        if len(parts) >= 3 and ".bundle" in parts[2]:
            k["kind"] = "bundle"
            name = parts[2]
            b = k["_b"].setdefault(name, {"hash": name.split(".")[0], "tmp": ".tmp." in name, "functions": set(),
                                          "has_e5": False, "payload_bytes": 0, "files": 0})
            if len(parts) >= 5 and not parts[4].endswith(".e5"):
                b["functions"].add(parts[4])
            if not e.is_dir:
                b["files"] += 1
                b["payload_bytes"] += e.size
                b["has_e5"] |= parts[-1].endswith(".e5")
    for k in keys.values():
        k["bundles"] = [{**b, "functions": sorted(b["functions"])} for b in k.pop("_b").values()]
    bundles = [b for k in keys.values() for b in k["bundles"]]
    return {"keys": keys, "totals": {
        "keys": len(keys), "milhash": sum(k["kind"] == "milhash" for k in keys.values()),
        "bundles": len(bundles), "bundles_with_payload": sum(b["payload_bytes"] > 0 and not b["tmp"] for b in bundles),
        "tmp_bundles": sum(b["tmp"] for b in bundles)}}


def diff(before: dict, after: dict) -> dict:
    a, b = before["keys"], after["keys"]
    lost = []
    for key, kb in a.items():
        now = {x["hash"]: x for x in b.get(key, {"bundles": []})["bundles"]}
        for x in kb["bundles"]:
            if x.get("payload_bytes", 0) > 0 and now.get(x["hash"], {}).get("payload_bytes", 0) == 0:
                lost.append({"key": key, "bundle": x["hash"], "functions": x["functions"],
                             "bytes_before": x["payload_bytes"]})
    return {"added_keys": sorted(set(b) - set(a)), "removed_keys": sorted(set(a) - set(b)), "payload_lost": lost}


def load(src: str) -> dict:
    p = Path(src).expanduser()
    return summarize(from_dir(p) if p.is_dir() else from_devicectl(p))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "summarize":
        print(json.dumps(load(args[0]), indent=1, sort_keys=True))
    elif cmd == "diff":
        print(json.dumps(diff(load(args[0]), load(args[1])), indent=1))
    else:
        raise SystemExit("usage: e5_listing.py summarize <json|dir> | diff <before> <after>")
```

- [ ] **Step 4: Run the tests, then check the parser against the real evidence.**

Run: `uv run --exact --group convert pytest tests/test_e5_listing.py -v` → 3 passed.
Run: `uv run --exact --group convert python -m scripts.probes.e5_listing summarize ~/egcm-evidence-raw/2026-10-27/listing-now.json | tail -8`
Expected totals: `keys 26, milhash 13, bundles 10, bundles_with_payload 0, tmp_bundles 1` (this reproduces N1).
Run it on `~/Library/Caches/probe/com.apple.e5rt.e5bundlecache` and check it against N2/N3.

- [ ] **Step 5: Commit.**

```bash
git add scripts/probes/__init__.py scripts/probes/e5_listing.py tests/test_e5_listing.py
git commit -m "feat(probes): e5_listing summarises and diffs Core ML E5 caches from the phone or the Mac"
```

