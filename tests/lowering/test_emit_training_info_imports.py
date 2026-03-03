from onnx import helper, TensorProto
from src.lowering.training_info_emit import emit_training_info
from src.graph_context import GraphContext


def test_emit_training_info_honors_imported_initializers():
    ctx = GraphContext(name="fuse")
    # simulate an imported initializer "Imported_W"
    init = helper.make_tensor(name="Imported_W", data_type=TensorProto.FLOAT, dims=[2,2], vals=[0.0,0.0,0.0,0.0])
    ctx.initializers["Imported_W"] = init
    ctx.value_types["Imported_W"] = {"scalar": "f32", "dims": [2,2]}
    ctx.defined_values.add("Imported_W")

    # create a node that looks like an optimizer producing an update: "Imported_W.opt"
    n = helper.make_node("Adam", ["Imported_W", "Imported_W.grad"], ["Imported_W.opt"], name="Imported_Adam")
    ctx.nodes.append(n)

    grad_summary = {"opt_updates": {"W": "Imported_W.opt"}, "optimizer_nodes": ["Imported_Adam"]}

    emit_training_info(ctx, grad_summary)

    assert getattr(ctx, "_training_info", None)
    ti = ctx._training_info[-1]
    # the update_binding should reference the actual model initializer name (prefixed)
    # and map it to the stripped algorithm output (e.g., "W.opt").
    found = None
    for e in ti.update_binding:
        if e.key == "Imported_W":
            found = e.value
    assert found == "W.opt"
