import os
from pathlib import Path
import json
import onnx
from src.parser import fuse_parser
from src.cli.cli_helpers import export_onnx_from_ast


def test_cli_embed_flag_controls_output(tmp_path):
    # prepare a simple .bin and .fuse
    b = tmp_path / "weights.bin"
    import numpy as np
    arr = np.array([1.0, 2.0], dtype=np.float32)
    with open(b, "wb") as f:
        f.write(arr.tobytes())

    fuse_file = tmp_path / "m.fuse"
    from src.util.project_version import get_project_version
    version = get_project_version()
    fuse_file.write_text(
        f"""
@fuse {version}
@opset onnx 18
@version {version}
@domain jupyter.cookbook
const W: f32[2] = @import("weights.bin")
node apply(x: f32[2]) -> f32[2] {{ y = MatMul(x, W) y }}
"""
    )

    ast = fuse_parser.parse(fuse_file.read_text(), filename=str(fuse_file))

    # externalized output (default)
    out_paths = export_onnx_from_ast(ast, source_file=str(fuse_file), out_dir=str(tmp_path), embed_external_data=False)
    assert out_paths, "no output path returned"
    model_path = out_paths[0]
    m = onnx.load(model_path)
    inits = list(m.graph.initializer)
    w = [i for i in inits if i.name.endswith("W") or i.name == "W"][0]
    # Should reference external_data (since not embedded).
    # Note: some ONNX loaders will automatically load external_data into
    # the initializer raw_data; accept either form but ensure metadata points
    # to the external file when externalized.
    if len(w.external_data) == 0:
        md = {p.key: p.value for p in m.metadata_props}
        assert "external_files" in md and md.get("external_files")
    else:
        assert len(w.external_data) > 0

    # embedded output
    out_paths2 = export_onnx_from_ast(ast, source_file=str(fuse_file), out_dir=str(tmp_path), embed_external_data=True)
    model_path2 = out_paths2[0]
    m2 = onnx.load(model_path2)
    inits2 = list(m2.graph.initializer)
    w2 = [i for i in inits2 if i.name.endswith("W") or i.name == "W"][0]
    assert len(w2.external_data) == 0
    assert w2.raw_data and len(w2.raw_data) > 0
