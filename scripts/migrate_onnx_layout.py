"""Utilities to migrate a flat ONNX artifact layout into the structured
domain/version layout produced by `src.io.path_utils.artifact_path_for`.

This module intentionally keeps a small, well-tested public surface:

- migrate(base, apply=False) -> List[Tuple[Path, Path]]

When `apply=False` the function acts as a dry-run and returns the planned
moves without modifying the filesystem. When `apply=True` the files will be
moved to their target locations (creating directories as needed).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import onnx

from src.io.path_utils import artifact_path_for


def migrate(base: str | Path, apply: bool = False) -> List[Tuple[Path, Path]]:
    """Scan `base` for ONNX files and compute moves into the structured layout.

    Parameters
    - base: base directory containing (flat) onnx files
    - apply: when True perform the filesystem move operations

    Returns: list of (src_path, dst_path) tuples for all moved/planned files
    """
    base = Path(base)
    moves: List[Tuple[Path, Path]] = []

    if not base.exists():
        return moves

    for p in base.rglob("*.onnx"):
        # Skip files that are already under a nested domain folder (heuristic):
        # if the file's parent is not the base directory itself we still consider
        # moving it - the path_utils will place it deterministically. This
        # keeps the implementation simple and idempotent.
        try:
            model = onnx.load(str(p))
        except Exception:
            # Skip files that cannot be loaded as ONNX
            continue

        try:
            dst = Path(artifact_path_for(model=model, base=str(base), flat=False))
        except Exception:
            # If we cannot compute a target path (missing domain etc.) skip
            continue

        # If source and destination are identical, nothing to do
        if p.resolve() == dst.resolve():
            continue

        moves.append((p, dst))

    # If apply, perform the filesystem changes
    if apply:
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Use replace to move atomically when possible
            src.replace(dst)

    return moves


__all__ = ["migrate"]
