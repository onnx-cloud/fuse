#!/usr/bin/env python3
# flake8: noqa: E402,F821
import sys
from pathlib import Path

import numpy as np

import onnx

# Runner assumes ONNX model already exists at onnx/cookbook/l1_score.onnx
model_path = Path("onnx/cookbook/l1_score.onnx")
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

sess = None
sample = np.array([1.0, 0.0, 0.0], dtype=np.float32)
try:
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    inp_name = sess.get_inputs()[0].name
    outs = sess.run(None, {inp_name: sample})
    print("ONNX Runtime output:", outs[0])
except Exception:
    from onnx.reference import ReferenceEvaluator

    sess = ReferenceEvaluator(m)
    inp_name = m.graph.input[0].name
    outs = sess.run(None, {inp_name: sample})
    print("ReferenceEvaluator output:", outs[0])

# Cleanup helper if created earlier
try:
    TMP.unlink()
except Exception:
    pass

if __name__ == "__main__":
    pass

# Load and run model
model_path = outdir / "l1_score.onnx"
from onnx.reference import ReferenceEvaluator

import onnx

# The fuse exporter sometimes writes into a directory; handle both cases.
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
sample = np.array([1.0, 0.0, 0.0], dtype=np.float32)
outs = sess.run(None, {input_name: sample})
print("Output:", outs[0])

# Cleanup
try:
    TMP.unlink()
except Exception:
    pass

if __name__ == "__main__":
    pass
