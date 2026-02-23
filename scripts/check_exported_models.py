#!/usr/bin/env python3
"""Load each exported ONNX model and run a small numeric check using
onnx.reference.ReferenceEvaluator to ensure it executes end-to-end.

The script will feed zeros of appropriate dtypes/shapes to every non-initializer
graph input and run the model once. Any failure raises an exception (exit 1).
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

# Prefer running via the project's virtualenv Python if available
try:
    import os
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parents[1]
    _venv_py = _here / ".venv" / "bin" / "python"
    if _venv_py.exists():
        try:
            if _Path(sys.executable).resolve() != _venv_py.resolve():
                os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
        except Exception:
            pass
except Exception:
    pass

import numpy as np

import onnx
from onnx import TensorProto

_NUMPY_FROM_ONNX = {
    TensorProto.FLOAT: np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.FLOAT16: np.float16,
    TensorProto.BFLOAT16: np.float16,
    TensorProto.INT8: np.int8,
    TensorProto.INT16: np.int16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.UINT8: np.uint8,
    TensorProto.UINT16: np.uint16,
    TensorProto.UINT32: np.uint32,
    TensorProto.UINT64: np.uint64,
    TensorProto.BOOL: np.bool_,
}


def _input_shape_dim(dim):
    # Return a concrete int for a dimension: prefer explicit dim_value, fall back to 1
    try:
        v = dim.dim_value
        if v and v > 0:
            return int(v)
    except Exception:
        pass
    return 1


def _tensor_dtype(elem_type):
    return _NUMPY_FROM_ONNX.get(elem_type, np.float32)


def _feed_for_input(value_info, initializer_names):
    name = value_info.name
    if name in initializer_names:
        return None
    tt = value_info.type.tensor_type
    shape = [_input_shape_dim(d) for d in tt.shape.dim]
    dtype = _tensor_dtype(tt.elem_type)
    return name, np.zeros(shape, dtype=dtype)


def check_model(model_path: str):
    print(f"Checking {model_path}")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    try:
        from onnx.reference import ReferenceEvaluator
    except Exception as e:
        raise RuntimeError(f"ReferenceEvaluator unavailable: {e}")

    sess = ReferenceEvaluator(model)

    init_names = {t.name for t in model.graph.initializer}
    feed = {}
    for vi in model.graph.input:
        pair = _feed_for_input(vi, init_names)
        if pair is None:
            continue
        name, arr = pair
        feed[name] = arr

    if not feed:
        print(
            f"No non-initializer inputs to feed for {model_path}; running with empty feed"
        )
        outputs = sess.run(None, {})
    else:
        outputs = sess.run(None, feed)

    if not outputs:
        raise RuntimeError(
            f"Model {model_path} produced no outputs during numeric check"
        )
    print(f"OK: {model_path} -> produced {len(outputs)} outputs")


if __name__ == "__main__":
    # Prefer checking the explicit export list created by export_cookbook.sh
    exports = Path("onnx/exports.txt")
    files = []
    if exports.exists():
        files = [
            line.strip()
            for line in exports.read_text().splitlines()
            if line.strip()
        ]
    else:
        files = glob.glob("onnx/**/*.onnx", recursive=True)

    if not files:
        print("No ONNX models found to check", file=sys.stderr)
        raise SystemExit(1)

    error = False
    for f in files:
        try:
            check_model(f)
        except Exception as e:
            print(f"[FAIL] {f}: {e}", file=sys.stderr)
            error = True
    if error:
        raise SystemExit(1)
    print("All exported models passed numeric checks")
