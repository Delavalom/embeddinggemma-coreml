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
