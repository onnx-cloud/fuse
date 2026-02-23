import subprocess
import sys
from pathlib import Path

import pytest

from src import metrics


def test_safe_infer_shapes_subprocess_failure(monkeypatch, tmp_path):
    # Create a minimal ONNX model file to pass to safe_infer_shapes via compute
    # We'll monkeypatch subprocess.run to simulate a crashing inference child
    class FakeProc:
        def __init__(self):
            self.returncode = 1
            self.stdout = ""
            self.stderr = "crash"

    def fake_run(*a, **k):
        return FakeProc()

    # Build a tiny model by lowering a simple fuse example (use existing helper)
    p = tmp_path / "m.fuse"
    p.write_text("""@fuse 0.7
@opset onnx 18
@version 0.0.0
@domain example
node id(x: f32[1]) -> f32[1] { x }
""")

    monkeypatch.setattr("subprocess.run", fake_run)

    # Should not raise despite subprocess failure; should return metrics dict with error absent
    m = metrics.compute_metrics_for_file(str(p))
    # We still expect a metrics dict with ops/graphs keys present
    assert isinstance(m, dict)
    assert "ops" in m
    assert "graphs" in m
