"""Helpers for schema-driven type-constraint inference.

This module provides small utilities that parse an ONNX ``OpSchema`` to map
type-parameters (e.g., ``T``) to concrete input types and to infer output
Fuse-style types such as ``{'scalar': 'f32', 'dims': [...]}``.
"""

import re
from typing import Any, Dict, List, Optional


def _extract_type_tokens(type_str: str) -> List[str]:
    """Return uppercase-like type tokens from a schema type string.

    e.g., ``"T1"`` or ``"T"`` from strings like ``"T"`` or ``"seq(T)"``.
    """
    if not type_str:
        return []
    return re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", str(type_str))


def map_type_params_from_inputs(
    schema, input_types: List[Optional[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """Map schema type-parameter names to concrete Fuse input types.

    Args:
        schema: an ONNX ``OpSchema`` or an object with an ``inputs`` attr.
        input_types: list of Fuse-style type dicts (or ``None``) matching the
            inputs.

    Returns:
        Mapping from type-parameter token (e.g., ``'T'``) to a Fuse-style
        type dict.
    """
    param_map: Dict[str, Dict[str, Any]] = {}
    for idx, inp_schema in enumerate(getattr(schema, "inputs", []) or []):
        if idx >= len(input_types):
            break
        inp_type = input_types[idx] or {}
        tstr = (
            getattr(inp_schema, "typeStr", None)
            or getattr(inp_schema, "type_str", "")
            or ""
        )
        for tok in _extract_type_tokens(tstr):
            if inp_type.get("scalar"):
                param_map[tok] = {
                    "scalar": inp_type.get("scalar"),
                    "dims": list(inp_type.get("dims") or []),
                }
    return param_map


def infer_output_from_schema(
    schema, input_types: List[Optional[Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """Infer an output Fuse-style type using schema bindings.

    Returns the first matching binding when an output references a type
    parameter that we were able to map from inputs; otherwise ``None``.
    """
    try:
        param_map = map_type_params_from_inputs(schema, input_types)
        if not param_map:
            return None
        for out_schema in getattr(schema, "outputs", []) or []:
            tstr = (
                getattr(out_schema, "typeStr", None)
                or getattr(out_schema, "type_str", "")
                or ""
            )
            for tok in _extract_type_tokens(tstr):
                if tok in param_map:
                    return param_map[tok]
    except Exception:
        # Conservative: never raise from this helper; keep lowering resilient.
        return None
    return None
