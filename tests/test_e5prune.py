"""egcm.e5prune: deletes large, old bundles created since start, keeping the newest large one."""

import os
import time

from egcm import e5cache, e5prune


def test_prunes_old_large_bundles_but_keeps_the_newest(tmp_path, monkeypatch):
    root = tmp_path / "python3" / "com.apple.e5rt.e5bundlecache" / "25G83"
    root.mkdir(parents=True)
    monkeypatch.setattr(e5cache, "CACHE_ROOTS", (root.parent,))
    (root / "preexisting").mkdir()
    (root / "preexisting" / "w").write_bytes(b"x" * 3000)
    p = e5prune.BundlePruner(min_mb=0.001, min_age_s=60)

    def bundle(name, age, size):
        d = root / name
        d.mkdir()
        (d / "w").write_bytes(b"x" * size)
        t = time.time() - age
        os.utime(d, (t, t))
        return d

    old1, old2, recent = bundle("a", 600, 3000), bundle("b", 300, 3000), bundle("c", 10, 3000)
    newest, small = bundle("d", 5, 3000), bundle("e", 600, 10)
    p.prune_once()
    assert not old1.exists() and not old2.exists()  # large, older than min_age_s, not the newest
    assert recent.exists() and newest.exists()  # too young / the newest large bundle
    assert small.exists() and (root / "preexisting").exists()  # below min_mb / not created by this process
