#!/usr/bin/env python3
"""
control_flow_smoke.py
Parse each control-flow smoke example and export the lowered ONNX model(s).
"""
import sys
from pathlib import Path
import onnx

from src.cli.helpers import parse_fuse_file
from src.lowering import FuseLowerer

if __name__ == "__main__":
    root = Path(__file__).parent
    fuse_file = root / "control_flow_smoke.fuse"
    ast = parse_fuse_file(str(fuse_file))
    fl = FuseLowerer()
    model = fl.lower(ast)
    out = root / "control_flow_smoke.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, str(out))
    print(f"exported {out}")
