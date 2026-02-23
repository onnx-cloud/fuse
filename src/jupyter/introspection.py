"""Introspection and completion helpers.

These functions return structured JSON arrays or dicts that a client (e.g., LSP
or custom frontend) can consume for completions, op names, attribute
suggestions and docstrings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any


def _load_ops_from_json() -> List[str]:
    p = Path(__file__).resolve().parents[2] / "ONNX_OPS.json"  # src/jupyter/introspection.py -> src -> fuse root
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        # ONNX_OPS.json is a list of dicts with 'name' field
        if isinstance(data, list):
            return sorted([op['name'] for op in data if isinstance(op, dict) and 'name' in op])
        # Fallback: if it's a dict, use keys
        if isinstance(data, dict):
            return sorted(list(data.keys()))
    except Exception:
        return []
    return []


_ops_cache = _load_ops_from_json()


def list_symbols(user_ns: dict) -> List[str]:
    """Return sorted list of symbol names present in the user's namespace."""
    return sorted(k for k in user_ns.keys())


def list_ops() -> List[str]:
    """Return a list of known op names (best-effort from ONNX_OPS.json)."""
    return _ops_cache


def op_attributes(op_name: str) -> List[Dict[str, Any]]:
    """Return a small list of attribute suggestions for an op.

    If ONNX is available, try to extract the operator schema for richer
    attribute names/types/docs. Otherwise fall back to a small static list.
    """
    try:
        import onnx
        try:
            # Try to get the latest schema for this operator
            schema = onnx.defs.get_schema(op_name, 1)
            attrs = []
            for a in schema.attributes:
                attrs.append({"name": a.name, "type": str(a.type), "doc": a.description})
            if attrs:
                return attrs
        except Exception:
            # get_schema may raise if not found or version mismatch; ignore
            pass
    except Exception:
        pass
    # Fallback
    return [
        {"name": "axis", "type": "int", "doc": "Axis along which to operate"},
        {"name": "keepdims", "type": "bool", "doc": "Keep reduced dims"},
    ]


def op_doc(op_name: str) -> str:
    try:
        import onnx
        try:
            schema = onnx.defs.get_schema(op_name, 1)
            return schema.doc or f"No detailed doc available for {op_name}."
        except Exception:
            pass
    except Exception:
        pass
    return f"No detailed doc available for {op_name} in this lightweight view."