import json
from typing import Any, Dict, List, Optional, Tuple

import onnx
from onnx import ModelProto


def _read_asserts_from_model(model: ModelProto) -> List[str]:
    for mp in model.metadata_props:
        if mp.key == "fuse.asserts":
            try:
                v = json.loads(mp.value)
                if isinstance(v, list):
                    return list(v)
            except Exception:
                # fallback: if stored as a plain string, return single
                return [mp.value]
    return []


def _model_input_shapes(model: ModelProto) -> Dict[str, List[Optional[int]]]:
    out = {}
    for vi in model.graph.input:
        name = vi.name
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            # ONNX uses dim_value==0 for unspecified/dynamic in some workflows;
            # treat any explicit integer dim_value as the integer (including 0).
            if hasattr(d, "dim_value") and d.dim_value is not None:
                dims.append(int(d.dim_value))
            else:
                dims.append(None)
        out[name] = dims
    return out


def _make_eval_context(model: ModelProto) -> Dict[str, Any]:
    shapes = _model_input_shapes(model)

    def shape(name: str):
        if name in shapes:
            return shapes[name]
        # try strip quotes
        return shapes.get(str(name).strip("'\""), [])

    def dim(name: str, i: int):
        s = shape(name)
        if i < 0 or i >= len(s):
            raise IndexError("dim index out of range")
        return s[i]

    ctx = {"shape": shape, "dim": dim}
    return ctx


def check_model_asserts(
    model: ModelProto, allow_unevaluable: bool = False
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Check assertions recorded in model metadata.

    Returns (success, details) where details is a list of dicts with keys:
      - expr: the original assertion text
      - status: 'true' | 'false' | 'unevaluable' | 'error'
      - message: optional diagnostic

    If any assertion evaluates to False, success is False.
    If an assertion is unevaluable and allow_unevaluable is False, success is False.
    """
    asserts = _read_asserts_from_model(model)
    if not asserts:
        return True, []

    ctx = _make_eval_context(model)
    details: List[Dict[str, Any]] = []
    overall_ok = True
    for a in asserts:
        try:
            # Evaluate in restricted globals; only allowed helpers present
            val = eval(a, {"__builtins__": {}}, ctx)
            if bool(val):
                details.append({"expr": a, "status": "true"})
            else:
                details.append({"expr": a, "status": "false"})
                overall_ok = False
        except Exception as e:
            details.append(
                {"expr": a, "status": "unevaluable", "message": str(e)}
            )
            if not allow_unevaluable:
                overall_ok = False
    return overall_ok, details


# Small CLI-friendly wrapper
def check_model_file(
    path: str, allow_unevaluable: bool = False
) -> Tuple[bool, List[Dict[str, Any]]]:
    m = onnx.load(path)
    return check_model_asserts(m, allow_unevaluable=allow_unevaluable)
