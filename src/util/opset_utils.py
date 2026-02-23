"""ONNX opset helpers.

Provides a single helper to compute deterministic opset imports for ModelProto
building and centralizes the SAFE_MAX_OPSET cap.
"""
from __future__ import annotations

from typing import Dict, Iterable, List


SAFE_MAX_OPSET = 23


def compute_opset_imports(default_opset: int | str, extra_opsets: Dict[str, int] | None = None, safe_max: int = SAFE_MAX_OPSET):
    """Return a list of (domain, version) tuples suitable for creating opset ids.

    - Ensures versions are capped at `safe_max`.
    - Ensures deterministic ordering: core opset ('') first, then sorted domains.
    - Accepts int or numeric string for `default_opset`.

    Note: returning tuples avoids an import-time dependency on ONNX helper
    and keeps this utility testable without heavy ONNX imports.
    """
    if default_opset is None:
        raise ValueError("default_opset must be provided")
    try:
        core = int(default_opset)
    except Exception:
        core = int(str(default_opset))
    if core > safe_max:
        core = safe_max

    tuples: List[tuple[str, int]] = [("", core)]

    extra = extra_opsets or {}
    for domain in sorted(extra.keys()):
        try:
            v = int(extra[domain])
        except Exception:
            v = int(str(extra[domain]))
        if v > safe_max:
            v = safe_max
        tuples.append((domain, v))

    return tuples
