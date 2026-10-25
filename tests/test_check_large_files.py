from scripts.check_large_files import MB, tracked_sizes, violations


def test_limits():
    assert violations({"a.json": 5 * MB, "fixtures/images/x.jpg": 2 * MB}) == []
    assert violations({"swift/w.bin": 5 * MB + 1}) == ["swift/w.bin: 5.0 MB > 5 MB"]
    assert violations({"fixtures/images/x.jpg": 2 * MB + 1}) == ["fixtures/images/x.jpg: 2.0 MB > 2 MB"]
    assert violations({"fixtures/images.jsonl": 3 * MB}) == []  # only files under fixtures/images/


def test_this_repo_is_within_limits():
    sizes = tracked_sizes()
    assert "gates.toml" in sizes
    assert violations(sizes) == []
