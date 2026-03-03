"""Declarative registry for ONNX operator lowering functions."""
from typing import Callable, Dict, Tuple, Optional

# key is (op_type, domain); value is (lowerer_func, min_opset)
_LOWERERS: Dict[Tuple[str, str], Tuple[Callable, int]] = {}


def onnx_op(op_type: str, domain: str = "", min_opset: int = 21):
    """Decorator to register a lowering function.

    Lowering implementations should be pure functions taking an
    ``OpsLowerer`` instance as the first argument, followed by the usual
    ``call, ctx, env, types`` parameters (see ``_lower_onnx_call`` for
    reference).  This allows handlers to access helper methods such as
    ``_lower_onnx_call`` for generic fallback behaviour.
    ``min_opset`` indicates the minimum opset for which this lowerer is
    valid.  The registrar will ignore entries when the current opset is
    lower than the requirement.
    """

    def decorator(func: Callable):
        _LOWERERS[(op_type, domain)] = (func, min_opset)
        return func

    return decorator


def get_lowerer(op_type: str, domain: str = "", opset: int = 21) -> Optional[Callable]:
    """Return the registered lowerer for ``op_type`` in ``domain`` if one exists
    and the provided ``opset`` meets the minimum requirement.

    Returns ``None`` when no suitable lowerer is found.
    """

    entry = _LOWERERS.get((op_type, domain))
    if entry and opset >= entry[1]:
        return entry[0]
    # fallback: try builtin domain if domain-specific entry is missing
    if domain and _LOWERERS.get((op_type, "")) and opset >= _LOWERERS[(op_type, "")][1]:
        return _LOWERERS[(op_type, "")][0]
    return None


def registered_ops() -> Dict[Tuple[str, str], int]:
    """Return a copy of registered operators with their minimum opset."""
    return {k: v[1] for k, v in _LOWERERS.items()}

import warnings

def validate_registry():
    """Check that all expected operators are registered.
    
    If lowerer modules fail to import, their operators won't be registered.
    This validates that key ones are present.
    """
    EXPECTED_OPS = {
        ("Cast", ""),  # from convert
        ("Add", ""),
        # Add basic ops we expect to always have
    }
    
    missing = EXPECTED_OPS - set(_LOWERERS.keys())
    if missing:
        warnings.warn(f"Missing operator lowerers: {missing}. Some lowering passes might fail.", UserWarning)
