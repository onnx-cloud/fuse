import onnx
from src.parser import fuse_parser
from src.cli.cli_helpers import export_onnx_from_ast


def test_inline_flag_affects_export(tmp_path):
    fuse_file = tmp_path / "f.fuse"
    fuse_file.write_text(
        """@opset onnx 18
@domain ex
fn foo(x: f32) -> f32 { Add(x, x) }
model m(x: f32) -> f32 { foo(x) }
"""
    )
    ast = fuse_parser.parse(fuse_file.read_text())
    # default (no inline) should produce a FunctionProto
    out1 = export_onnx_from_ast(ast, source_file=str(fuse_file), out_dir=str(tmp_path))
    m1 = onnx.load(out1[0])
    assert any(fn.name == "foo" for fn in m1.functions)
    # with inline=True the function is expanded
    out2 = export_onnx_from_ast(ast, source_file=str(fuse_file), out_dir=str(tmp_path), inline=True)
    m2 = onnx.load(out2[0])
    assert len(m2.functions) == 0
    assert any(n.op_type == "Add" for n in m2.graph.node)
