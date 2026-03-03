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
    """Ensure the directory for `path` exists (creates parents as needed).
    
    Args:
        path (str): The directory path or file path whose parent directory should exist.
        
    Returns:
        None
        
    Raises:
        OSError: If directory creation fails due to permissions or other OS errors.
    """
    p = Path(path)
    d = p if p.is_dir() else p.parent
    d.mkdir(parents=True, exist_ok=True)


def write_binary_atomic(data: bytes, path: str) -> None:
    """Write bytes to `path` atomically by writing to a temp file and replacing.

    This avoids partially-written files being visible to other processes.
    
    Args:
        data (bytes): The binary data to write.
        path (str): The target file path.
        
    Returns:
        None
        
    Raises:
        OSError: If writing to the temporary file or replacing the target file fails.
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

    Args:
        entries (List[Dict[str, str]]): List of entries defining `src` and `dest` file mapping.
        out_dir (str): Destination directory.

    Returns:
        None

    Raises:
        OSError: If destination directory creation fails.
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
