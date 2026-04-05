from src.parser import fuse_parser
from src.lowering import FuseLowerer
from src.lowering.training_checks import validate_training_info
from tests.test_utils import project_fuse_version
FUSE_DECL = f"@fuse {project_fuse_version()}\n"


def test_emit_training_info_creates_traininginfo():
    src = FUSE_DECL + """
    @train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]
    @training { optimizer = Adam, lr = 1e-3 }
    node m(x: f32[2]) -> f32[2] { return MatMul(W, x) }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)
    # Ensure we emitted a TrainingInfoProto entry and it validates
    assert len(model.training_info) >= 1
    validate_training_info(model)  # should not raise


def test_grad_output_passes_through_producer():
    src = """
    @train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]
    @training { optimizer = Adam, lr = 1e-3 }
    node m(x: f32[2]) -> f32[2] {
        y = MatMul(W, x)
        y
    }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)

    # Output should include the canonical gradient name
    outs = [o.name for o in model.graph.output]
    assert any(name.endswith("W.grad") for name in outs)

    # There should NOT be a synthetic Identity node whose output is W.grad
    identity_produces_grad = any(
        (n.op_type == "Identity" and any(o.endswith("W.grad") for o in n.output))
        for n in model.graph.node
    )
    assert not identity_produces_grad


def test_emit_training_info_includes_optimizer_state_outputs():
    from onnx import helper, TensorProto
    from src.lowering.training_info_emit import emit_training_info
    from src.graph_context import GraphContext

    ctx = GraphContext(name="fuse")
    init = helper.make_tensor(name="W", data_type=TensorProto.FLOAT, dims=[2, 2], vals=[1.0, 0.0, 0.0, 1.0])
    ctx.initializers["W"] = init
    ctx.value_types["W"] = {"scalar": "f32", "dims": [2, 2]}

    node = helper.make_node(
        "Adam",
        ["W", "W.grad", "lr"],
        ["W.opt", "W.m", "W.v"],
        name="AdamOptimizer",
    )
    try:
        node.domain = "ai.onnx.preview.training"
    except Exception:
        pass
    ctx.nodes.append(node)

    grad_summary = {"opt_updates": {"W": "W.opt"}, "optimizer_nodes": ["AdamOptimizer"]}
    emit_training_info(ctx, grad_summary)

    assert getattr(ctx, "_training_info", None)
    ti = ctx._training_info[-1]
    alg = ti.algorithm
    output_names = {o.name for o in alg.output}

    assert "W.opt" in output_names
    assert "W.m" in output_names
    assert "W.v" in output_names
