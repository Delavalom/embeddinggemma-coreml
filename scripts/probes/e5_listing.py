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
