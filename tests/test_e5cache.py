"""egcm.e5cache: switching functions deletes the cache bundles created since start, never older ones."""

from egcm import e5cache


def test_purges_only_new_bundles(tmp_path, monkeypatch):
    root = tmp_path / "python3" / "com.apple.e5rt.e5bundlecache" / "25G83"
    old = root / "OLD"
    old.mkdir(parents=True)
    monkeypatch.setattr(e5cache, "CACHE_ROOTS", (root.parent,))
    created = []

    class FakeModel:
        def __init__(self, path, compute_units=None, function_name=None):
            d = root / f"B_{function_name}"
            d.mkdir()
            (d / "w.bin").write_bytes(b"x" * 1000)
            created.append(d)

    monkeypatch.setattr(e5cache.ct.models, "CompiledMLModel", FakeModel)
    p = e5cache.PurgingPackage(tmp_path / "M.mlmodelc", e5cache.ct.ComputeUnit.CPU_ONLY)
    p.function("a")
    p.function("a")  # same function: no reload, no purge
    assert [d.exists() for d in created] == [True]
    p.function("b")
    assert [d.exists() for d in created] == [False, True] and old.exists()
    p.close()
    assert not created[1].exists() and old.exists()
    assert p.purged_mb == 0.002
