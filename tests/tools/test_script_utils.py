import sys
from pathlib import Path

from scripts import script_utils


def test_bootstrap_adds_repo_root_to_sys_path(tmp_path, monkeypatch):
    # Ensure the repo root is not already at front
    root = str(script_utils.repo_root())
    if root in sys.path:
        sys.path.remove(root)

    script_utils.bootstrap_script()
    assert sys.path[0] == root or root in sys.path


def test_repo_root_location():
    # repo_root() should point to the directory containing 'scripts'
    root = script_utils.repo_root()
    assert (root / "scripts").is_dir()
