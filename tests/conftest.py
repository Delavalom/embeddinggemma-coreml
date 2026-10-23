import subprocess

import pytest


def git(root, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=root, check=True, capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repo shaped like the real one, with one commit."""
    root = tmp_path / "repo"
    for rel, text in {
        "egcm/paths.py": "X = 1\n",
        "egcm/model/x.py": "M = 1\n",
        "egcm/stage.py": "S = 1\n",
        "stages/s0_reference/run.py": "# group: reference\n",
        "stages/s1_rewrite/run.py": "# group: convert\n",
        "gates.toml": "[s0_reference]\nmin_a = 0.5\n[s1_rewrite]\nmin_a = 0.5\n",
        "pyproject.toml": "[project]\nname = 'x'\n",
        "uv.lock": "version = 1\n",
        "fixtures/MANIFEST.sha256": "abc  img.jpg\n",
        "README.md": "hi\n",
    }.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def commit(root, message="change"):
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)
