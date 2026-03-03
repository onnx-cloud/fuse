"""Helpers to construct and validate Model metadata emitted into ONNX.

This centralizes '@fuse' validation, version emission, and sanitization of
user-supplied metadata for serialization into ModelProto.metadata_props.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.util.project_version import get_project_version


def _parse_ver(s: str) -> tuple[int, int, int]:
    parts = [int(p) for p in s.split(".") if p.isdigit()]
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def build_emitted_metadata(model_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate `@fuse` metadata and return a dict ready to serialize.

    Raises RuntimeError on missing or invalid @fuse declarations.
    """
    mm = model_metadata or {}

    # authoritative project version
    pkg_fuse_version = get_project_version()
    if pkg_fuse_version is None:
        raise RuntimeError("Unable to determine installed 'fuse' package version")

    declared_fuse = mm.get("fuse")
    if declared_fuse is None:
        if model_metadata is None:
            # Default to installed package version for snippets/tests without explicit @fuse
            declared_fuse = pkg_fuse_version
        else:
            raise RuntimeError(
                f"Missing top-level '@fuse' declaration. Every .fuse file must include a top-level '@fuse <MAJOR.MINOR[.PATCH]>' metadata entry and it must be less than or equal to the installed/runtime repository version ({pkg_fuse_version})."
            )

    if not re.match(r"^\d+(?:\.\d+){0,2}$", str(declared_fuse)):
        raise RuntimeError(
            f"Invalid @fuse version '{declared_fuse}': expected 'MAJOR.MINOR' or 'MAJOR.MINOR.PATCH' (numeric)."
        )

    declared_t = _parse_ver(str(declared_fuse))
    pkg_t = _parse_ver(str(pkg_fuse_version))

    if declared_t > pkg_t:
        # In tests/CI we may auto-inject a higher @fuse value to avoid
        # brittle failures. Honor an explicit env var to allow that case
        # while keeping strict behavior by default.
        import os

        if str(os.environ.get("FUSE_AUTO_INJECT", "")).lower() in ("1", "true", "yes"):
            # permissive for test runs
            pass
        else:
            raise RuntimeError(
                f"@fuse version '{declared_fuse}' but we support <= {pkg_fuse_version}. "
                f"Please upgrade your installed 'fuse' package or use {pkg_fuse_version}."
            )

    emitted: Dict[str, Any] = {}
    # Preserve declared @fuse in emitted metadata while also recording the
    # runtime/package version separately so tooling can inspect both.
    emitted["fuse"] = str(declared_fuse)
    emitted["fuse_runtime"] = str(pkg_fuse_version)
    emitted["version"] = str(mm.get("version", pkg_fuse_version))
    emitted["created_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # Merge user-supplied metadata (override computed keys) but avoid proto messages
    try:
        from google.protobuf.message import Message as _ProtoMessage
    except Exception:
        _ProtoMessage = None

    for k, v in mm.items():
        # Skip protobuf Message objects
        if _ProtoMessage and isinstance(v, _ProtoMessage):
            continue
        # Do not allow user metadata to override computed keys
        if str(k) in {"fuse", "version", "created_at"}:
            continue
        if isinstance(v, dict):
            clean = {}
            for kk, vv in v.items():
                if _ProtoMessage and isinstance(vv, _ProtoMessage):
                    clean[kk] = f"<{type(vv).__name__}>"
                else:
                    clean[kk] = vv
            emitted[str(k)] = clean
        else:
            emitted[str(k)] = v

    return emitted
