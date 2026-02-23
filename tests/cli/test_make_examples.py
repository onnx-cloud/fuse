import os
import subprocess
from pathlib import Path


def test_make_examples_runs_and_exports(tmp_path):
    """Ensure `make examples` compiles golden examples and exports artifacts."""
    env = os.environ.copy()
    # Ensure ensure-venv sees an active venv (it only checks VIRTUAL_ENV presence)
    env["VIRTUAL_ENV"] = str(Path.cwd() / ".venv")

    proc = subprocess.run(["make", "examples"], capture_output=True, text=True, env=env)

    # Expect the target to succeed (Makefile bug was fixed) and export examples
    assert proc.returncode == 0, f"Expected zero exit code, got {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    output = proc.stdout + proc.stderr
    # Confirm the exporter processed at least the first example and continued
    assert "Processing examples/golden/arithmetic.fuse" in output
    assert "Processing examples/golden/clip.fuse" in output
    assert "✅ All examples exported successfully." in output
