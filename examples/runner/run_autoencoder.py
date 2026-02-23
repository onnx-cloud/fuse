#!/usr/bin/env python3
# flake8: noqa: E402,F821
import sys
from pathlib import Path

import numpy as np

import onnx

# Runner assumes ONNX model already exists at onnx/cookbook/autoencoder.onnx
model_path = Path("onnx/cookbook/autoencoder.onnx")
if model_path.is_dir():
    found = next(model_path.rglob("*.onnx"), None)
    if not found:
        print(
            f"Exported directory {model_path} contains no .onnx",
            file=sys.stderr,
        )
        sys.exit(1)
    model_path = found
if not model_path.exists():
    print(f"asserted ONNX model not found: {model_path}", file=sys.stderr)
    sys.exit(1)

m = onnx.load(str(model_path))
onnx.checker.check_model(m)

sample = np.array([1.0, 2.0, 3.0], dtype=np.float32)
try:
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: sample})[0]
    print("ONNX Runtime output:", out)
except Exception:
    from onnx.reference import ReferenceEvaluator

    sess = ReferenceEvaluator(m)
    inp_name = m.graph.input[0].name
    out = sess.run(None, {inp_name: sample})[0]
    print("ReferenceEvaluator output:", out)

print("Done")

# Load and run model
import onnx

m = onnx.load(model_path)
onnx.checker.check_model(m)

sample = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# Try onnxruntime first
try:
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: sample})[0]
    print("ONNX Runtime output:", out)
except Exception:
    # Fallback to ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator

        sess = ReferenceEvaluator(m)
        inp_name = m.graph.input[0].name
        out = sess.run(None, {inp_name: sample})[0]
        print("ReferenceEvaluator output:", out)
    except Exception as e:
        print("No runtime available to execute model:", e)
        sys.exit(1)

print("Done")
