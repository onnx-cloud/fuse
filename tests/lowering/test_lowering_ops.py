import json
from pathlib import Path
import pytest

import onnx
from onnx import TensorProto
from src.lowering import FuseLowerer
from src.parser import fuse_parser


def test_lower_matmul():
    src = """
    node mm(a: f32[2,3], b: f32[3,4]) -> f32[2,4] {
      return a @ b
    }
    """
    ast = fuse_parser.parse(src)
    model = FuseLowerer().lower(ast)
    onnx.checker.check_model(model)
    assert any(n.op_type == "MatMul" for n in model.graph.node)


def test_lower_elementwise_broadcast():
    src = """
    node mul(x: f32[2], y: f32) -> f32[2] {
      return x * y
    }
    """
    ast = fuse_parser.parse(src)
    model = FuseLowerer().lower(ast)
    onnx.checker.check_model(model)
    # Expect a Mul node; scalar operand may be unsqueezed -> still Mul present
    assert any(n.op_type == "Mul" for n in model.graph.node)


def test_lower_cast_node():
    src = """
    node c(x: i64) -> f32 {
      return Cast<f32>(x)
    }
    """
    ast = fuse_parser.parse(src)
    model = FuseLowerer().lower(ast)
    onnx.checker.check_model(model)
    casts = [n for n in model.graph.node if n.op_type == "Cast"]
    assert casts, "Cast node not emitted"
    # Verify the attribute `to` corresponds to FLOAT
    found = False
    for n in casts:
        for a in n.attribute:
            if a.name == "to" and a.i == int(TensorProto.FLOAT):
                found = True
    assert found


def test_lower_sequence_output():
    # Use an existing example that returns a sequence (LoopBlock)
    p = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "advanced"
        / "loop_block.fuse"
    )
    src = p.read_text()
    ast = fuse_parser.parse(src)
    model = FuseLowerer().lower(ast)
    onnx.checker.check_model(model)
    outs = model.graph.output
    assert any(o.type.sequence_type is not None for o in outs)


def test_fused_const_emits_external_files(tmp_path: Path):
    src = (
        '@domain ns\n'
        'const big: f32[2] = @import("data.bin")\n'
        'node use() -> f32[2] {\n'
        '  return big\n'
        '}\n'
    )
    # create referenced data file
    data = tmp_path / "data.bin"
    data.write_bytes(b"\x00\x01")
    ast = fuse_parser.parse(src)
    model = FuseLowerer().lower(
        ast, source_file=str(tmp_path / "someone.fuse")
    )
    # model metadata should include external_files JSON (don't run onnx.checker
    # here because the lowered model refers to external files by name which
    # are resolved/copy-embedded during runtime packaging/tests).
    md = {kv.key: kv.value for kv in model.metadata_props}
    assert "external_files" in md
    ext = json.loads(md["external_files"])
    assert ext and ext[0]["init_name"] == "big"
    assert str(data) in ext[0]["src"]
    # MISSING-006: TODO - verify model initializers actually match external data file
    # (Currently only checks metadata reference; should verify data integrity)


def test_quantize_annotation_emits_quantize_nodes():
    src = """
    @quantize("int8")
    node q(x: f32) -> f32 {
      return x
    }
    """
    ast = fuse_parser.parse(src)
    model = FuseLowerer().lower(ast)
    onnx.checker.check_model(model)
    # Find QuantizeLinear or Cast nodes (emitted when @quantize applied)
    assert any(
        n.op_type in ("QuantizeLinear", "Cast") for n in model.graph.node
    )


def test_scan_num_scan_inputs():
    from src.util.project_version import get_project_version
    version = get_project_version()
    src = f"""\
@fuse {version}
@opset onnx 18
@domain scantest
node sc(seq: list[f32], st: f32) -> f32 {{
    # simple scan: propagate state unmodified
    return scan(seq, st) {{
        new = state_in
        return new
    }}
}}
"""
    ast = fuse_parser.parse(src)
    model = FuseLowerer().lower(ast)
    onnx.checker.check_model(model)
    scans = [n for n in model.graph.node if n.op_type == "Scan"]
    assert scans, "no Scan node emitted"
    attr = next((a for a in scans[0].attribute if a.name == "num_scan_inputs"), None)
    assert attr is not None
    assert attr.i == 1

    # second scenario: two sequence inputs should count as two scan inputs
    src2 = f"""\
@fuse {version}
@opset onnx 18
@domain scantest
node sc2(seq1: list[f32], seq2: list[f32], st: f32) -> f32 {{
    return scan(seq1, seq2, st) {{
        new = state_in
        return new
    }}
}}
"""
    ast2 = fuse_parser.parse(src2)
    model2 = FuseLowerer().lower(ast2)
    scans2 = [n for n in model2.graph.node if n.op_type == "Scan"]
    assert scans2, "no Scan in second scenario"
    attr2 = next((a for a in scans2[0].attribute if a.name == "num_scan_inputs"), None)
    assert attr2 is not None
    assert attr2.i == 2


def test_loop_injects_condition():
    from src.util.project_version import get_project_version
    version = get_project_version()
    src = f"""\
@fuse {version}
@opset onnx 18
@domain looptest
node lt(count: i64) -> f32 {{
    out = loop(count, true, 0.0) {{
        new = state_in
        return new
    }}
    return out
}}
"""
    ast = fuse_parser.parse(src)
    model = FuseLowerer().lower(ast)
    onnx.checker.check_model(model)
    # verify that the Loop body's first output is bool by inspecting the graph
    loop_node = next(n for n in model.graph.node if n.op_type == "Loop")
    body = next(a for a in loop_node.attribute if a.name == "body").g
    assert body.output and body.output[0].type.tensor_type.elem_type == int(onnx.TensorProto.BOOL)


def test_if_missing_condition_raises():
    from src.lowering import FuseLowerer
    from src.graph_context import GraphContext
    from src.lowering.utils import LoweringError

    fl = FuseLowerer()
    ops = fl.ops_lowerer
    ctx = GraphContext()
    with pytest.raises(LoweringError):
        ops._lower_if_call({"call": "If"}, ctx, {}, {})

