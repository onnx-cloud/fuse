"""Utilities to deterministically compute artifact save paths for ONNX models.

Provides `artifact_path_for` which maps model metadata -> filesystem path
using a domain-based hierarchy by default. See `todo/VERSION.md` for design
notes.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
import warnings


def _get_domain_from_meta(meta: dict) -> str | None:
    if not isinstance(meta, dict):
        return None
    dom = meta.get("domain")
    if dom is None and "module" in meta:
        warnings.warn("metadata key 'module' is deprecated; use 'domain' instead", DeprecationWarning)
        dom = meta.get("module")
    return dom

import onnx


_safe_re = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_segment(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # Remove dangerous path traversals and separators
    s = s.replace("..", "")
    s = s.replace("/", "-")
    s = s.replace(os.path.sep, "-")
    # Collapse whitespace and unsafe chars to '-'
    s = _safe_re.sub("-", s)
    # Trim repeated dashes
    s = re.sub(r"-+", "-", s)
    s = s.strip("-._ ")
    if not s:
        return ""
    return s


def _version_folder(version: Optional[str]) -> str:
    if not version:
        return "unversioned"
    # Prefer major.minor if available: e.g., 1.2.3 -> v1.2
    m = re.match(r"^(\d+)\.(\d+)", str(version))
    if m:
        return f"v{m.group(1)}.{m.group(2)}"
    # Fallback: sanitize and prefix with 'v'
    v = _sanitize_segment(version)
    return f"v{v}" if v else "unversioned"


def artifact_path_for(
    model: onnx.ModelProto | None = None,
    *,
    base: str = "./tmp/onnx",
    flat: bool = False,
    variant: Optional[str] = None,
    model_meta: Optional[dict] = None,
) -> str:
    """Compute a deterministic artifact path for an ONNX model.

    Parameters
    - model: Optional ModelProto. If provided, metadata is read from it.
    - base: Base directory for artifacts (default: ./tmp/onnx)
    - flat: When True, preserve legacy flat layout and return "<base>/<name>.onnx"
    - variant: Optional variant string appended to name as `--{variant}`.
    - model_meta: Optional dict of metadata keys to use instead of `model`.

    Returns: an absolute or relative path string under `base`.

    Raises:
    - ValueError when domain is missing (and not flat).
    """
    if model_meta is None and model is not None:
        meta = {kv.key: kv.value for kv in getattr(model, "metadata_props", [])}
    else:
        meta = model_meta or {}

    # Determine name: prefer provided graph/name metadata, else fallback to model.graph.name
    name = meta.get("name") or meta.get("title")
    if not name and model is not None and getattr(model, "graph", None) and getattr(model.graph, "name", None):
        name = model.graph.name
    if not name:
        name = "model"
    name = _sanitize_segment(name)
    # If the name is a qualified dotted name (e.g., 'examples.golden.jepa.encode')
    # prefer the last local component for file naming so emitted artifact names
    # remain concise and stable for downstream tests and users.
    if "." in name:
        name = name.split(".")[-1]

    # Flat legacy mode: base/<name>.onnx (append variant if present)
    if flat:
        fname = name
        if variant:
            fname = f"{fname}--{_sanitize_segment(variant)}"
        return str(Path(base) / f"{fname}.onnx")

    # Domain: prefer explicit metadata keys (canonical 'domain', legacy 'module').
    domain = _get_domain_from_meta(meta) or meta.get("domain")
    if not domain and model is not None:
        domain = getattr(model, "domain", None)
    if not domain:
        raise ValueError("model domain is required for structured artifact layout")

    # Normalize domain into folder parts by splitting on '.' or '/'
    parts = [p for p in re.split(r"[\./]", str(domain)) if p]
    parts = [_sanitize_segment(p) for p in parts]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("invalid model domain")

    # Version folder
    version = meta.get("version")
    version_folder = _version_folder(version)

    # Variant handling
    fname = name
    if variant:
        fname = f"{fname}--{_sanitize_segment(variant)}"

    # Ensure no path traversal in parts
    for p in parts + [version_folder, fname]:
        if ".." in p:
            raise ValueError("invalid path component")

    final = Path(base)
    final = final.joinpath(*parts)
    final = final / version_folder
    return str(final / f"{fname}.onnx")
