"""egcm.e5exit: a process deletes the Core ML E5 bundles it created when it exits, also on failure; run_stage too."""

import os
import signal

import pytest

from egcm import e5exit
from egcm.stage import run_stage


def cache(tmp_path):
    root = tmp_path / "python3" / "com.apple.e5rt.e5bundlecache"
    (root / "25G83" / "OLD").mkdir(parents=True)
    return root


def make_bundle(root, name):
    d = root / "25G83" / name
    d.mkdir()
    (d / "w.bin").write_bytes(b"x" * 2000)
    return d


def test_deletes_only_bundles_created_inside(tmp_path):
    root = cache(tmp_path)
    with e5exit.ExitPurge((root,)) as p:
        new = make_bundle(root, "NEW")
    assert not new.exists() and (root / "25G83" / "OLD").exists()
    assert p.deleted_mb == 0.002


def test_deletes_on_an_exception(tmp_path):
    root = cache(tmp_path)
    with pytest.raises(RuntimeError), e5exit.ExitPurge((root,)):
        new = make_bundle(root, "NEW")
        raise RuntimeError("stage failed")
    assert not new.exists()


def test_deletes_on_sigterm(tmp_path):
    root = cache(tmp_path)
    with pytest.raises(SystemExit) as e, e5exit.ExitPurge((root,)):
        new = make_bundle(root, "NEW")
        os.kill(os.getpid(), signal.SIGTERM)
    assert e.value.code == 128 + signal.SIGTERM and not new.exists()
    assert signal.getsignal(signal.SIGTERM) is not e5exit._sigterm  # the previous handler is back


def test_missing_cache_is_fine(tmp_path):
    with e5exit.ExitPurge((tmp_path / "nothing",)) as p:
        pass
    assert p.deleted_mb == 0.0


def test_run_stage_purges_after_a_failed_stage(repo, tmp_path, monkeypatch):
    root = cache(tmp_path)
    monkeypatch.setattr(e5exit, "CACHE_ROOTS", (root,))
    created = []

    def fails(r):
        created.append(make_bundle(root, "STAGE"))
        raise ValueError("boom")

    with pytest.raises(SystemExit):
        run_stage("s0_reference", fails, root=repo)
    assert created and not created[0].exists() and (root / "25G83" / "OLD").exists()
