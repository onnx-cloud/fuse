from onnx import helper
from src.lowering.training_info_emit import emit_training_info
from src.graph_context import GraphContext, fuse_dtype_to_onnx


def test_emit_training_info_respects_scalar_dtype_mapping():
    ctx = GraphContext(name="fuse")
    # create a node that produces an optimizer update (non-initializer input)
    n = helper.make_node("Adam", ["x", "x.grad"], ["x.opt"], name="AdamNode")
    ctx.nodes.append(n)

    # assign a non-default scalar type to the input
    ctx.value_types["x"] = {"scalar": "i64", "dims": [2]}

    grad_summary = {"opt_updates": {"x": "x.opt"}, "optimizer_nodes": ["AdamNode"]}

    emit_training_info(ctx, grad_summary)

    assert getattr(ctx, "_training_info", None)
    ti = ctx._training_info[-1]

    # find algorithm input for 'x' and check elem_type
    in_map = {vi.name: vi for vi in ti.algorithm.input}
    assert "x" in in_map
    assert int(in_map["x"].type.tensor_type.elem_type) == int(fuse_dtype_to_onnx("i64"))

    # find algorithm output for 'x.opt' and check elem_type
    out_map = {vi.name: vi for vi in ti.algorithm.output}
    assert "x.opt" in out_map
    assert int(out_map["x.opt"].type.tensor_type.elem_type) == int(fuse_dtype_to_onnx("i64"))
