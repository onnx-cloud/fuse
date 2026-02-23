"""Top-level CLI IO helpers (avoid `src.cli` package name collision).

This mirrors the implementation in `src/cli/io.py` but exposes a module
importable as `src.cli_io` to avoid entrypoint name clashes.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


def ensure_dir_exists(path: str) -> None:
    p = Path(path)
    d = p if p.is_dir() else p.parent
    d.mkdir(parents=True, exist_ok=True)


def write_binary_atomic(data: bytes, path: str) -> None:
    target = Path(path)
    ensure_dir_exists(str(target))
    fd, tmp = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, str(target))
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def copy_external_files(entries: List[Dict[str, str]], out_dir: str) -> None:
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
