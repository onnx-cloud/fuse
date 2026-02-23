import json
from pathlib import Path

# NOTE: The implementation `src.inspector.inspect_model` does not exist yet; these
# tests are intentionally written first (TDD) and will fail until the feature is
# implemented.
from src.inspector import inspect_model


def test_inspect_writes_core_artifacts(tmp_path):
    # Accept either committed fixture `onnx/golden.onnx` (legacy) or generated
    # golden model under `tmp/onnx/golden.onnx` (current default for `make gold`).
    candidates = ["tmp/onnx/golden.onnx", "onnx/golden.onnx"]
    onnx_model = None
    for c in candidates:
        p = Path(c)
        if p.exists():
            onnx_model = p
            break
    assert onnx_model is not None, "missing golden model fixture: checked tmp/onnx/golden.onnx and onnx/golden.onnx"

    out_dir = tmp_path / "golden.onnx/"

    # Call the core API. Parameters mirror the proposed CLI flags.
    res = inspect_model(
        str(onnx_model),
        out_dir=str(out_dir),
        dot=True,
        render=True,
        interactive=False,
        plots=True,
        filter_re=None,
        force=True,
        dry_run=False,
    )

    # Expect a non-empty listing of paths written
    assert res, "expected inspect_model to return list of paths"

    # Basic artifact expectations
    ast_file = out_dir / "ast.json"
    fuse_file = out_dir / "model.fuse"
    meta_file = out_dir / "metadata.json"
    dot_file = out_dir / "graph.dot"

    for f in (ast_file, fuse_file, meta_file, dot_file):
        assert f.exists(), f"expected {f} to be written"

    # Rendering: either an image or an error file should be present
    svg_file = out_dir / "graph.svg"
    svg_err = out_dir / "graph.svg.error.txt"
    assert svg_file.exists() or svg_err.exists(), "expected graph.svg or graph.svg.error.txt"

    # Validate JSON contents
    data = json.loads(ast_file.read_text(encoding="utf-8"))
    assert isinstance(data, (list, dict))

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert isinstance(meta, dict)
    # Prefer keys we expect in a summary metadata file
    assert any(k in meta for k in ("shapes", "ops", "params"))
