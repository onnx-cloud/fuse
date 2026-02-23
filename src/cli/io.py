"""Filesystem helpers used by CLI commands.

Provides small utilities that encapsulate file and directory operations and
make them easier to test and mock.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


def ensure_dir_exists(path: str) -> None:
    """Ensure the directory for `path` exists (creates parents as needed)."""
    p = Path(path)
    d = p if p.is_dir() else p.parent
    d.mkdir(parents=True, exist_ok=True)


def write_binary_atomic(data: bytes, path: str) -> None:
    """Write bytes to `path` atomically by writing to a temp file and replacing.

    This avoids partially-written files being visible to other processes.
    """
    target = Path(path)
    ensure_dir_exists(str(target))
    fd, tmp = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        # atomic replace
        os.replace(tmp, str(target))
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def copy_external_files(entries: List[Dict[str, str]], out_dir: str) -> None:
    """Copy external files described by `entries` into `out_dir`.

    Each entry is expected to be a dict with keys:
      - src: source path
      - dest: destination filename under out_dir
      - init_name: optional (ignored here)

    Warnings are printed to stderr on failure but the operation is best-effort
    and does not raise for individual file copy failures.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    for ent in entries:
        src = ent.get("src")
        dest = ent.get("dest")
        if not src or not dest:
            print(
                f"Warning: skipping invalid external entry {ent}",
                file=sys.stderr,
            )
            continue
        try:
            shutil.copy2(src, str(outp / dest))
        except Exception as e:
            print(
                f"Warning: failed to copy external file {src} -> {dest}: {e}",
                file=sys.stderr,
            )
