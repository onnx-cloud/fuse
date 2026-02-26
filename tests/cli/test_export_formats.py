import os
from pathlib import Path
import pytest
pytest.importorskip("lark")
import onnx
import json
from src.parser import fuse_parser
from src.cli.cli_helpers import export_onnx_from_ast
import src.cli.cli_helpers as ch


def _write_dummy(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("dummy")


def test_export_flags_invoke_converters(tmp_path, monkeypatch):
    # prepare a simple .fuse
    fuse_file = tmp_path / "m.fuse"
    from src.util.project_version import get_project_version
    version = get_project_version()
    fuse_file.write_text(
        f"""
@fuse {version}
@opset onnx 18
@version {version}
@domain jupyter.cookbook
node apply(x: f32[2]) -> f32[2] {{ y = MatMul(x, x) y }}
"""
    )

    ast = fuse_parser.parse(fuse_file.read_text(), filename=str(fuse_file))

    # Monkeypatch conversion helpers to avoid optional heavy deps
    def fake_tf(onnx_model, base, dest_dir):
        p = str(Path(dest_dir) / f"{base}.tf" )
        _write_dummy(p + "/SAVEDMODEL")
        return p

    def fake_tfl(saved, base, dest_dir):
        p = str(Path(dest_dir) / f"{base}.tflite")
        _write_dummy(p)
        return p

    def fake_pt(onnx_model, base, dest_dir):
        p = str(Path(dest_dir) / f"{base}.pt")
        _write_dummy(p)
        return p

    monkeypatch.setattr(ch, "_export_tf_model_from_onnx", fake_tf)
    monkeypatch.setattr(ch, "_export_tflite_from_saved", fake_tfl)
    monkeypatch.setattr(ch, "_export_pt_from_onnx_model", fake_pt)

    out_paths = export_onnx_from_ast(ast, source_file=str(fuse_file), out_dir=str(tmp_path), tf=True, tfl=True, pt=True)
    # First path must be ONNX
    assert out_paths and out_paths[0].endswith(".onnx")
    # Ensure subsequent paths include our dummy files
    assert any(p.endswith(".tf") for p in out_paths)
    assert any(p.endswith(".tflite") for p in out_paths)
    assert any(p.endswith(".pt") for p in out_paths)
