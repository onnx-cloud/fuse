from pathlib import Path

from src import cli_helpers


def test_export_multiple_models(tmp_path):
    fuse_file = Path("examples/golden/jepa.fuse")
    assert fuse_file.exists(), "jepa.fuse fixture must exist"
    ast = cli_helpers.parse_fuse_file(str(fuse_file))
    out_dir = tmp_path
    models = cli_helpers.export_onnx_from_ast(ast, source_file=str(fuse_file), out_dir=str(out_dir))
    assert models, "expected emitted ONNX models"
    names = sorted(Path(p).stem for p in models)
    expected = {"jepa_train", "jepa_encode", "jepa_predict", "jepa_target_encode"}
    assert expected.issubset(set(names)), f"expected {expected} in emitted model names, got {names}"
