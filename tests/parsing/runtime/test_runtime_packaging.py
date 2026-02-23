import json
from pathlib import Path

from src.lowering import FuseLowerer
from src.ort_web import build_ort_web_bundle
from src.parser import fuse_parser


def _build_ns_model(tmp_path: Path):
    """Helper to build a namespaced model for testing."""
    src = """
    @domain ns
    model m(x: f32) -> f32 {
      return x
    }
    """
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    model = lowerer.lower(ast, source_file=str(tmp_path / "example.fuse"))
    return model


def test_self_namespace_canonical(tmp_path: Path):
    model = _build_ns_model(tmp_path)

    in_names = [i.name for i in model.graph.input if i.name]
    out_names = [o.name for o in model.graph.output if o.name]

    # Expect qualified names like 'ns.m.x'
    assert any(n == "ns.m.x" for n in in_names)
    assert any(n == "ns.m.x" for n in out_names)


def test_ort_asset_provider_mock(tmp_path: Path):
    # Create a minimal ONNX model for packaging
    model = _build_ns_model(tmp_path)

    # Create fake vendor assets
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    js = vendor / "ort-wasm.js"
    wasm = vendor / "ort-wasm.wasm"
    js.write_text("// dummy js")
    wasm.write_bytes(b"dummy wasm")

    out_dir = tmp_path / "out"
    build_ort_web_bundle(model, str(out_dir), vendor_dir=str(vendor))

    # Assert outputs exist
    assert (out_dir / "model.onnx").exists()
    assert (out_dir / "ort-wasm.js").exists()
    assert (out_dir / "ort-wasm.wasm").exists()
    manifest = json.loads((out_dir / "model.json").read_text())

    assert manifest["runtime"]["js"]["path"] == "ort-wasm.js"
    assert manifest["runtime"]["wasm"]["path"] == "ort-wasm.wasm"
