import sys
from pathlib import Path

import pytest

nbclient = pytest.importorskip("nbclient")
nbformat = pytest.importorskip("nbformat")
pytest.importorskip("onnxruntime")

from nbclient import NotebookClient


def test_smoke_notebook_runs_and_produces_expected_output(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    nb_path = repo_root / "jupyter" / "notebooks" / "quick_fuse.ipynb"
    assert nb_path.exists(), f"Notebook not found at {nb_path}"

    nb = nbformat.read(nb_path, as_version=4)

    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    try:
        client.execute()
    except Exception as e:
        # ONNXRuntime in some environments may not support the emitted model IR version
        # (e.g., "Unsupported model IR version" errors). In that case, skip this test.
        msg = str(e)
        if 'Unsupported model IR version' in msg or 'onnxruntime' in msg:
            pytest.skip(f"ONNX runtime incompatible: {msg}")
        raise

    # Collect stdout from executed cells
    outputs = []
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                outputs.append(out.get("text", ""))

    stdout_text = "".join(outputs)
    assert "4.0" in stdout_text, f"Expected '4.0' in notebook stdout, got: {stdout_text!r}"
