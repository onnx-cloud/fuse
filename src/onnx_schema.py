"""ONNX schema helpers.

These are used to provide nicer errors than `onnx.checker` when the cognitive compiler emits
an operator that doesn't exist in the selected opset.
"""

from __future__ import annotations

import onnx


def require_op_schema(
    op_type: str, opset: int, domain: str = ""
) -> "onnx.defs.OpSchema":
    """Return schema or raise ValueError with a helpful message."""
    try:
        return onnx.defs.get_schema(op_type, int(opset), domain)
    except Exception as e:
        # Give a targeted error; callers can still run onnx.checker later.
        dom = "onnx" if domain == "" else domain
        msg = (
            f"Unknown operator '{op_type}' for domain '{dom}' "
            f"at opset {int(opset)}"
        )
        raise ValueError(msg) from e


def is_op_available(op_type: str, opset: int, domain: str = "") -> bool:
    try:
        onnx.defs.get_schema(op_type, int(opset), domain)
        return True
    except Exception:
        return False


def normalize_domain_and_op(op_type: str) -> tuple[str, str]:
    """Split optional `domain::OpType` form.

    Fuse grammar currently allows dots in identifiers, which is ambiguous for
    domain separation. We support an explicit `domain::OpType` convention.
    """
    if "::" in op_type:
        domain, op = op_type.split("::", 1)
        return domain, op
    return "", op_type
