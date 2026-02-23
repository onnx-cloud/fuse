import sys
from pathlib import Path

import pytest

nbclient = pytest.importorskip("nbclient")
nbformat = pytest.importorskip("nbformat")
pytest.importorskip("onnxruntime")

from nbclient import NotebookClient


def test_welcome_notebook_runs_and_verifies_setup(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    nb_path = repo_root / "jupyter" / "notebooks" / "welcome.ipynb"
    assert nb_path.exists(), f"Notebook not found at {nb_path}"

    nb = nbformat.read(nb_path, as_version=4)

    # Ensure our IPython extension (for the %%fuse magics) is loaded in the kernel
    load_cell = nbformat.v4.new_code_cell('from src.jupyter.ipython import load_ipython_extension\nload_ipython_extension(get_ipython())')
    nb['cells'].insert(0, load_cell)

    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    client.execute()

    # Collect stdout from executed cells
    outputs = []
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                outputs.append(out.get("text", ""))

    stdout_text = "".join(outputs)
    # Core environment checks should be present even if optional services (local server, LLM)
    # are not available in the test environment.
    assert "ONNX library" in stdout_text, f"Missing ONNX library check in stdout: {stdout_text!r}"
    assert "ONNX Runtime" in stdout_text, f"Missing ONNX Runtime check in stdout: {stdout_text!r}"
    assert "NumPy" in stdout_text, f"Missing NumPy check in stdout: {stdout_text!r}"
    assert "Fuse IPython magics" in stdout_text, f"Missing Fuse IPython magics check in stdout: {stdout_text!r}"
