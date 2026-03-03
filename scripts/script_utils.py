"""Shared helpers used by top-level scripts.

Most of the scripts under ``scripts/`` previously contained near-identical
boilerplate to 1) ensure the project root is on :data:`sys.path` and 2)
re-exec themselves using the virtualenv's Python interpreter when a `.venv` is
present.  Over time the boilerplate drifted and grew, leading to duplication and
minor inconsistencies across scripts.

This module centralizes that behavior so the individual script files can stay
shorter and easier to maintain.  It does **not** attempt to cover every aspect
of golden script environment logic (that script had additional
re-exec fallback when optional deps are missing) but the common bits are here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the absolute path of the repository root (parent of ``scripts/``)."""
    return Path(__file__).resolve().parents[1]


def ensure_repo_root_on_path() -> None:
    """Add the repository root to ``sys.path`` if it's not already present.

    This allows scripts invoked via ``python scripts/foo.py`` to import the
    local ``src`` package regardless of the current working directory.
    """
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def reexec_in_venv() -> None:
    """If a project virtualenv exists, re-exec the current program inside it.

    Looks for ``.venv/bin/python`` under the repository root.  If that binary
    exists and differs from ``sys.executable``, this function will ``os.execv``
    into the venv interpreter with the same argv.  Failures are silently
    ignored so callers need not wrap this call in a try/except.

    Typically this is called at the top of a script; if the exec succeeds the
    rest of the file is never executed under the original interpreter.
    """
    root = repo_root()
    venv_py = root / ".venv" / "bin" / "python"
    try:
        if venv_py.exists():
            # resolve to canonical paths to avoid false negatives due to symlinks
            if Path(sys.executable).resolve() != venv_py.resolve():
                os.execv(str(venv_py), [str(venv_py)] + sys.argv)
    except Exception:
        # best-effort only; if execv fails we continue with the current
        # interpreter rather than crash the script.
        pass


def bootstrap_script() -> None:
    """Perform the standard initialization for a top-level script.

    This is a convenience wrapper that adds the repo root to ``sys.path`` and
    then attempts to re-exec in the virtualenv.  It is idempotent and may be
    called multiple times safely.
    """
    ensure_repo_root_on_path()
    reexec_in_venv()
