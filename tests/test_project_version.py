import os
import pytest
from src.util.project_version import get_project_version


def test_env_override_returns_value():
    # Ensure env var override takes precedence
    prev = os.environ.get("FUSE_PROJECT_VERSION")
    try:
        os.environ["FUSE_PROJECT_VERSION"] = "9.9.9"
        assert get_project_version() == "9.9.9"
    finally:
        if prev is None:
            os.environ.pop("FUSE_PROJECT_VERSION", None)
        else:
            os.environ["FUSE_PROJECT_VERSION"] = prev


def test_installed_version_fallback():
    # If an installed 'fuse' package version is available use it; otherwise skip.
    prev = os.environ.get("FUSE_PROJECT_VERSION")
    try:
        os.environ.pop("FUSE_PROJECT_VERSION", None)
        # We accept either the repository's pyproject.toml version (when running
        # in-tree) or the installed package version. Just ensure we return a
        # non-empty string.
        gv = get_project_version()
        assert isinstance(gv, str) and gv
    finally:
        if prev is None:
            os.environ.pop("FUSE_PROJECT_VERSION", None)
        else:
            os.environ["FUSE_PROJECT_VERSION"] = prev
