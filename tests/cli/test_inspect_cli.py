from pathlib import Path

from src.cli import __init__ as cli_module


def test_cli_inspect_smoke_writes_files(tmp_path):
    """Smoke test for the CLI surface. Ensure `fuse inspect` runs and writes
    expected artifacts.

    Accept either committed fixture `onnx/golden.onnx` (legacy) or generated
    golden model under `tmp/onnx/golden.onnx` (current default for `make gold`).
    """
    # Accept any available ONNX model fixture under known locations so the
    # smoke test remains robust across different test environments.
    candidates = list(Path("tmp/onnx").rglob("*.onnx")) + list(Path("onnx").rglob("*.onnx"))
    onnx_model = None
    for p in candidates:
        if p.exists():
            onnx_model = p
            break
    assert onnx_model is not None, "missing ONNX model fixture under tmp/onnx or onnx/"

    out_dir = tmp_path / "out"
    argv = ["inspect", "-f", str(onnx_model), "-o", str(out_dir), "--dot"]

    rc = cli_module.main(argv)

    assert rc == 0, "inspect command should exit successfully"

    # Check core artifacts
    stem = Path(str(onnx_model)).stem
    ast_file = out_dir / stem / "ast.json"
    fuse_file = out_dir / stem / "model.fuse"
    dot_file = out_dir / stem / "graph.dot"
    meta_file = out_dir / stem / "metadata.json"

    for f in (ast_file, fuse_file, dot_file, meta_file):
        assert f.exists(), f"expected {f} to be written by inspect"
