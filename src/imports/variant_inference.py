"""Variant inference from ONNX model data types.

Infers variant names (e.g., 'int8', 'u8', 'fp32') from ONNX initializers
and tensor types. This ensures deterministic, unambiguous variant labeling
even when initializers have mixed dtypes.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import onnx
from onnx import TensorProto


def infer_variant_from_elem_type(elem_type: int) -> str:
    """Map ONNX elem_type to a canonical variant name.

    Args:
        elem_type: TensorProto enum value (e.g., TensorProto.INT8, TensorProto.UINT8)

    Returns:
        Variant name string such as 'int8', 'u8', 'fp32', etc.

    Raises:
        ValueError: if elem_type is unknown or unsupported.
    """
    mapping: Dict[int, str] = {
        TensorProto.FLOAT: "fp32",
        TensorProto.DOUBLE: "f64",
        TensorProto.INT8: "int8",
        TensorProto.UINT8: "u8",  # Critical: explicit U8 case to avoid ambiguity with INT8
        TensorProto.INT16: "int16",
        TensorProto.UINT16: "u16",
        TensorProto.INT32: "int32",
        TensorProto.UINT32: "u32",
        TensorProto.INT64: "int64",
        TensorProto.UINT64: "u64",
        TensorProto.BOOL: "bool",
        TensorProto.FLOAT16: "f16",
        TensorProto.BFLOAT16: "bf16",
    }

    result = mapping.get(int(elem_type))
    if result is None:
        raise ValueError(
            f"Unknown or unsupported elem_type: {elem_type}. "
            f"Supported types: {sorted(mapping.keys())}"
        )
    return result


def infer_variant_from_model(model: onnx.ModelProto) -> Optional[str]:
    """Infer a single canonical variant name from an ONNX model's initializers.

    Scans all initializers in the model's graph. If all initializers have
    the same dtype, returns the inferred variant. If dtypes are mixed or
    no initializers exist, returns None (caller should use explicit variant or default).

    Args:
        model: onnx.ModelProto to inspect

    Returns:
        Variant name if all initializers are the same type, None otherwise.
    """
    if not model.graph.initializer:
        return None

    elem_types: List[int] = []
    for init in model.graph.initializer:
        elem_types.append(int(init.data_type))

    # If all initializers have the same type, infer the variant
    unique_types = set(elem_types)
    if len(unique_types) == 1:
        elem_type = unique_types.pop()
        try:
            return infer_variant_from_elem_type(elem_type)
        except ValueError:
            # Unsupported type; let caller handle
            return None

    # Mixed dtypes; cannot infer a single variant
    return None


def infer_variant_from_model_metadata(model: onnx.ModelProto) -> Optional[str]:
    """Extract variant name from self-describing ONNX model metadata.

    Looks for a 'variant' key in the model's metadata_props.
    Returns the value if present, None otherwise.

    Args:
        model: onnx.ModelProto to inspect

    Returns:
        Variant name from metadata, or None if not found.
    """
    if not hasattr(model, "metadata_props") or not model.metadata_props:
        return None

    for prop in model.metadata_props:
        if prop.key == "variant":
            return prop.value

    return None
