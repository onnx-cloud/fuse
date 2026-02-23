"""Tests for scripts/update_onnx_ops.py

This test is intentionally opt-in: it will only run when the environment
variable UPDATE_ONNX_OPS=1 is set. Running it will update (or create)
`ONNX_OPS.json` at the repository root.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ONNX_OPS.json"
BACKUP = ROOT / "ONNX_OPS.json.bak-test"


@pytest.mark.skipif(os.getenv("UPDATE_ONNX_OPS") != "1", reason="opt-in test: set UPDATE_ONNX_OPS=1 to update ONNX_OPS.json")
def test_update_onnx_ops_creates_valid_json_and_contains_conv():
    # Back up existing file if present
    if OUT.exists():
        shutil.move(str(OUT), str(BACKUP))

    try:
        # Run the script (CLI path)
        subprocess.run([sys.executable, "-m", "scripts.update_onnx_ops", "--output", str(OUT)], check=True)

        assert OUT.exists(), "ONNX_OPS.json was not created"

        data = json.loads(OUT.read_text())
        assert isinstance(data, list), "ONNX_OPS.json should contain a list"

        names = {d["name"]: d for d in data}
        assert "Conv" in names, "Expected 'Conv' in generated ops"
        conv = names["Conv"]
        assert "kernel_shape" in conv.get("attributes", []), "Conv should have 'kernel_shape' attribute"

    finally:
        # restore backup
        if BACKUP.exists():
            if OUT.exists():
                OUT.unlink()
            shutil.move(str(BACKUP), str(OUT))
        else:
            # if no backup, leave the produced file in place (intentional update)
            pass
