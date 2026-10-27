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
