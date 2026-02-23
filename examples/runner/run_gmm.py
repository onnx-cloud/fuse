#!/usr/bin/env python3
# flake8: noqa: E402,F821
import sys
from pathlib import Path

import numpy as np

import onnx

# Runner assumes ONNX model already exists at onnx/cookbook/gmm.onnx
model_path = Path("onnx/cookbook/gmm.onnx")
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

sample = np.array([0.0, 0.0], dtype=np.float32)
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

from onnx.reference import ReferenceEvaluator

# Run
import onnx

model_path = outdir / "gmm.onnx"
if model_path.is_dir():
    found = next(model_path.rglob("*.onnx"), None)
    if not found:
        print(f"Exported directory {model_path} contains no .onnx")
        sys.exit(1)
    model_path = found

m = onnx.load(model_path)
onnx.checker.check_model(m)
sess = ReferenceEvaluator(m)
input_name = m.graph.input[0].name
sample = np.array([0.0, 0.0], dtype=np.float32)
outs = sess.run(None, {input_name: sample})
print("Posterior for sample", sample, "=>", outs[0])
