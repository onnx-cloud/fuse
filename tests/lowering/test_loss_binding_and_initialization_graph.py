import pytest
pytest.importorskip("onnx")
import onnx
from onnx import helper, TensorProto
from src.lowering.training_info_emit import emit_training_info
from src.graph_context import GraphContext


def test_loss_binding_and_initialization_graph():
    ctx = GraphContext(name="fuse")
    # create a model initializer and types
    init = helper.make_tensor(name="W", data_type=TensorProto.FLOAT, dims=[2,2], vals=[1.0,0.0,0.0,1.0])
    ctx.initializers["W"] = init
    ctx.value_types["W"] = {"scalar": "f32", "dims": [2,2]}
    ctx.value_types["loss"] = {"scalar": "f32", "dims": []}

    # create an optimizer node that also emits a 'loss' output
    n = helper.make_node("Adam", ["W", "W.grad"], ["W.opt", "loss"], name="MyAdam")
    ctx.nodes.append(n)

    grad_summary = {"opt_updates": {"W": "W.opt"}, "optimizer_nodes": ["MyAdam"]}

    emit_training_info(ctx, grad_summary)

    assert getattr(ctx, "_training_info", None)
    ti = ctx._training_info[-1]

    # initialization graph should be present (possibly empty)
    assert ti.initialization is not None

    # loss_binding should map model loss to algorithm output 'loss'
    found = False
    for e in ti.loss_binding:
        if e.key == "loss":
            found = True
            assert e.value == "loss"
    assert found
