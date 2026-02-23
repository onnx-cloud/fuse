"""ONNX opset helpers.

Centralizes opset discovery/validation so upgrading ONNX is low-touch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import onnx
from onnx import version_converter


def latest_onnx_opset() -> int:
    """Return the highest default-domain opset supported by the installed onnx package."""
    # onnx.defs.onnx_opset_version exists in all modern onnx releases.
    try:
        return int(onnx.defs.onnx_opset_version())
    except Exception:
        # Conservative fallback.
        return 18


def validate_opset_version(opset: int) -> int:
    try:
        opset_int = int(opset)
    except Exception as e:
        raise ValueError(f"Invalid opset value {opset!r}") from e

    if opset_int <= 0:
        raise ValueError(f"Invalid opset {opset_int}; must be >= 1")

    max_supported = latest_onnx_opset()
    if opset_int > max_supported:
        msg = (
            f"Requested opset {opset_int} is higher than installed "
            f"onnx supports ({max_supported}). "
            "Upgrade the 'onnx' package or lower the module @opset."
        )
        raise ValueError(msg)

    return opset_int


def get_model_default_opset(model: onnx.ModelProto) -> Optional[int]:
    for opset in model.opset_import:
        if opset.domain == "":
            return int(opset.version)
    return None


@dataclass(frozen=True)
class OpsetConversionError(RuntimeError):
    source_opset: Optional[int]
    target_opset: int
    message: str

    def __str__(self) -> str:
        src = (
            "unknown" if self.source_opset is None else str(self.source_opset)
        )
        return f"Cannot convert imported model opset {src} -> {self.target_opset}: {self.message}"


def convert_model_to_opset(
    model: onnx.ModelProto, target_opset: int
) -> onnx.ModelProto:
    """Convert model's default-domain opset to `target_opset`.

    Only converts the ONNX default domain. Extra domains are preserved as-is.
    """
    target = validate_opset_version(target_opset)
    src = get_model_default_opset(model)
    if src is None or src == target:
        return model

    try:
        return version_converter.convert_version(model, target)
    except Exception as e:
        raise OpsetConversionError(
            source_opset=src, target_opset=target, message=str(e)
        ) from e
