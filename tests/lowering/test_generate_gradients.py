from src.parser import fuse_parser
from src.lowering import FuseLowerer


def _out_names(model):
    return [o.name for o in model.graph.output]


def test_generate_gradients_adds_param_grads():
    from tests.test_utils import project_fuse_version
    FUSE_DECL = f"@fuse {project_fuse_version()}\n"
    src = FUSE_DECL + """
    @train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]
    node m(x: f32[2]) -> f32[2] { return MatMul(W, x) }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)
    outs = _out_names(model)
    assert any(name.endswith("W.grad") for name in outs)
    # There should also be a Gradient node producing the grad
    assert any(n.op_type == "Gradient" for n in model.graph.node)
    # Ensure the Gradient op is in the ONNX training preview domain
    assert any((n.op_type == "Gradient" and n.domain == "ai.onnx.preview.training") for n in model.graph.node)


def test_generate_gradients_respects_disabled_trainable():
    src = """
    weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]
    node m(x: f32[2]) -> f32[2] { return MatMul(W, x) }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)
    outs = _out_names(model)
    assert not any(name.endswith("W.grad") for name in outs)


def test_generate_gradients_exposes_loss_if_present():
    # Create a concrete loss node so it appears as a distinct internal value
    src = """
    node m(x: f32) -> f32 {
      y = x
      loss = Add(y, 1.0)
      return loss
    }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)
    outs = _out_names(model)
    # Loss should be exposed as a graph output
    assert any(name.endswith("loss") for name in outs)
