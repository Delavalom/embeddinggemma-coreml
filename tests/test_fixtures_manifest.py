import json

from egcm.fixtures import load_images, load_texts, render_manifest, verify_manifest, write_manifest
from egcm.paths import FIXTURES


def test_committed_fixtures_match_manifest():
    assert verify_manifest() == []


def test_fixture_rows():
    texts, images = load_texts(), load_images()
    assert [t["id"] for t in texts] == [f"t{i:03d}" for i in range(200)]
    assert sum(t["kind"] == "query" for t in texts) == 120
    assert len(images) == 200 and len({r["id"] for r in images}) == 200
    assert all(r["license"] in {"CC0", "Public domain"} for r in images)
    assert all((FIXTURES / r["file"]).is_file() for r in images)


def test_verify_manifest_reports_each_kind_of_mismatch(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"a")
    (tmp_path / "texts.jsonl").write_text(json.dumps({"id": "t000"}) + "\n")
    write_manifest(tmp_path)
    assert verify_manifest(tmp_path) == []
    assert render_manifest(tmp_path).splitlines()[0].endswith("  images/a.jpg")
    (tmp_path / "images" / "a.jpg").write_bytes(b"changed")
    (tmp_path / "images" / "b.jpg").write_bytes(b"b")
    (tmp_path / "texts.jsonl").unlink()
    assert verify_manifest(tmp_path) == [
        "missing: texts.jsonl", "not in manifest: images/b.jpg", "hash mismatch: images/a.jpg"]


CATEGORIES = {"people", "dogs", "cats", "food", "landscapes", "buildings", "documents", "cars", "screenshots", "memes",
              "blurry", "dark", "near_duplicates", "receipts"}


def test_fixture_images_are_plain_srgb_rgb_matching_jsonl():
    from PIL import Image

    for r in load_images():
        with Image.open(r["path"]) as im:
            assert im.mode == "RGB", r["id"]
            assert "icc_profile" not in im.info, f"{r['id']} carries an ICC profile"
            assert not im.getexif(), f"{r['id']} carries EXIF"
            assert im.size == (r["width"], r["height"]), r["id"]


def test_fixture_rows_are_consistent():
    rows = load_images()
    by_id = {r["id"]: r for r in rows}
    photos = {r["id"] for r in rows if r.get("title")}
    assert len(photos) == 100
    for r in rows:
        assert r["category"] in CATEGORIES, r["id"]
        assert r["file"].startswith(f"images/{r['id']}_{r['category']}."), r["id"]
        if r["category"] == "screenshots":
            assert r["kind"] in {"chat", "tweet", "settings"} and r["font"], r["id"]
        if "derived_from" in r:
            src = by_id[r["derived_from"]]
            assert r["derived_from"] in photos and r["license"] == src["license"], r["id"]
            assert r["source_url"] == src["source_url"], r["id"]
        if r["category"] in {"memes", "blurry", "dark", "near_duplicates"}:
            assert "derived_from" in r, r["id"]


def test_commons_sources_are_pinned():
    sources = json.loads((FIXTURES / "commons_sources.json").read_text())
    photos = [r for r in load_images() if r.get("title")]
    assert [s["title"] for s in sources] == [r["title"] for r in photos]
    for s in sources:
        assert s["license"] in {"CC0", "Public domain"}
        assert len(s["sha256"]) == 64 and s["sha1"] and s["timestamp"], s["title"]
        assert "@" not in json.dumps(s.get("download_url", "")), s["title"]
