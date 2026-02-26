import subprocess
import sys

import pytest

from scripts import gold


class DummyCompleted:
    def __init__(self, returncode=1, stdout="", stderr="err line1\nerr line2"):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_concise_failure(monkeypatch, capsys):
    # Simulate a failing subprocess.run for the 'test' step
    def fake_run(cmd, capture_output, text):
        assert "pytest" in " ".join(cmd)
        return DummyCompleted(returncode=2, stdout="", stderr="Traceback (most recent call last)\nAssertionError: fail")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = gold.main(["--step", "test"])
    assert rc != 0
    out = capsys.readouterr().out
    assert "ERROR: step 'test' failed" in out
    assert "Traceback (most recent call last)" in out
    assert "Run with --trace" in out


def test_trace_shows_full(monkeypatch, capsys):
    # Simulate a failing subprocess.call when --trace is used
    def fake_call(cmd):
        assert "pytest" in " ".join(cmd)
        return 3

    monkeypatch.setattr(subprocess, "call", fake_call)

    rc = gold.main(["--trace", "--step", "test"])
    assert rc != 0
    out = capsys.readouterr().out
    assert "--- Running: test" in out


def test_runs_golden_export(monkeypatch, capsys):
    # Simulate a successful subprocess.run for the 'golden-onnx' step
    def fake_run(cmd, capture_output, text):
        cmdstr = " ".join(cmd)
        assert "golden_onnx_export.py" in cmdstr or "scripts.golden_onnx_export" in cmdstr
        return DummyCompleted(returncode=0, stdout="exported\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = gold.main(["--step", "golden-onnx"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "✅ golden-onnx completed." in out
