"""Entry point for lowering operators.

The goal is to migrate from the current monolithic ``OpsLowerer`` to a
decorator-driven registry.  This package exposes a simple API for
registering and querying lowering functions.

The existing ``src/lowering/ops.py`` will gradually be refactored into
submodules under this package.
"""
from .registry import get_lowerer, onnx_op
# import submodules to ensure their registrations execute at import time
from . import elementwise  # noqa: F401
from . import convert  # noqa: F401

from ..ops import OpsLowerer  # backward-compatible export

__all__ = ["get_lowerer", "onnx_op", "OpsLowerer"]
from .registry import validate_registry
validate_registry()
