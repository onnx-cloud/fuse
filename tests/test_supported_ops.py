"""Tests for scripts/supported_ops.py

This test is opt-in: set UPDATE_OPS=1 to run and (re)generate `ONNX_OPS.json` at repo root.
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
OUT = ROOT / "OPS.json"
BACKUP = ROOT / "OPS.json.bak-test"


@pytest.mark.skipif(os.getenv("UPDATE_OPS") != "1", reason="opt-in test: set UPDATE_OPS=1 to update ONNX_OPS.json")
def test_supported_ops_creates_valid_json_and_has_expected_shape():
    # Back up existing file if present
    if OUT.exists():
        shutil.move(str(OUT), str(BACKUP))

    try:
        subprocess.run([sys.executable, "-m", "scripts.supported_ops", "--output", str(OUT)], check=True)

        assert OUT.exists(), "OPS.json was not created"

        data = json.loads(OUT.read_text())
        assert isinstance(data, list), "OPS.json should contain a list"

        if data:
            first = data[0]
            # check presence of keys
            for k in ("name", "domain", "since", "inputs", "attributes"):
                assert k in first, f"Expected key '{k}' in ops item"

            assert isinstance(first["attributes"], list) and all(isinstance(a, str) for a in first["attributes"])
            assert isinstance(first["inputs"], list)
            for inp in first["inputs"]:
                assert isinstance(inp, dict)
                assert isinstance(inp.get("name"), str)
                assert isinstance(inp.get("type"), str)
                assert isinstance(inp.get("optional"), bool)

        # check common op exists
        names = {d["name"]: d for d in data}
        assert "Conv" in names, "Expected 'Conv' in generated ops"

    finally:
        # restore backup
        if BACKUP.exists():
            if OUT.exists():
                OUT.unlink()
            shutil.move(str(BACKUP), str(OUT))
        else:
            # if no backup, leave the produced file in place (intentional update)
            pass


def test_collect_supported_ops_has_expected_shape():
    # non-opt-in unit test that verifies the in-memory collector shape and basic invariants
    from scripts import supported_ops

    ops = supported_ops.collect_supported_ops()
    assert isinstance(ops, list)
    if ops:
        first = ops[0]
        for k in ("name", "domain", "since", "inputs", "attributes"):
            assert k in first


def test_collect_supported_ops_sorted():
    from scripts import supported_ops

    ops = supported_ops.collect_supported_ops()
    pairs = [(o.get("domain", ""), o.get("name", "")) for o in ops]
    assert pairs == sorted(pairs)
