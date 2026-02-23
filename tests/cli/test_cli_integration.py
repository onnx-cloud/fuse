import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]


def run_cmd(args):
    return subprocess.call([sys.executable, "-m", "src.cli"] + args)


def test_cli_onnx_minimal(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    rc = run_cmd(
        ["compile", "-f", "examples/cookbook/golden_proof.fuse", "-o", str(out)]
    )
    assert rc == 0


def test_cli_run_minimal():
    rc = run_cmd(["run", "-f", "examples/cookbook/golden_proof.fuse"])
    assert rc == 0


def test_cli_golden_minimal():
    rc = run_cmd(["golden", "-f", "examples/cookbook/golden_proof.fuse"])
    assert rc == 0
