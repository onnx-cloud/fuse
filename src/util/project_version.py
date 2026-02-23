"""Authoritative project version helper.

Provides a single API `get_project_version()` used across the repo to determine
what the repository/package authoritative fuse version is. This centralizes
behavior and makes tests easier (use env var override `FUSE_PROJECT_VERSION`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def get_project_version() -> Optional[str]:
    """Return authoritative fuse project version or None.

    Resolution order:
      1. Environment override: `FUSE_PROJECT_VERSION` (testing/CI convenience)
      2. Search parent directories for `pyproject.toml` and return `project.version`.
      3. Fall back to `importlib.metadata.version('fuse')` when installed.

    Returns None when no version can be determined.
    """
    # 1) Environment override
    try:
        import os

        v = os.environ.get("FUSE_PROJECT_VERSION")
        if v:
            return str(v)
    except Exception:
        pass

    # 2) pyproject.toml in repository parents
    try:
        import tomllib
        p = Path(__file__).resolve()
        for parent in p.parents:
            py = parent / "pyproject.toml"
            if py.exists():
                try:
                    data = tomllib.loads(py.read_text(encoding="utf-8"))
                    ver = data.get("project", {}).get("version")
                    if ver:
                        return str(ver)
                except Exception:
                    return None
    except Exception:
        # tomllib may not be available or parsing failed—best-effort only
        pass

    # 3) installed package metadata
    try:
        import importlib

        return importlib.metadata.version("fuse")
    except Exception:
        return None
