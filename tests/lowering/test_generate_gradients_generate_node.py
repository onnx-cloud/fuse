from src.parser import fuse_parser
from src.lowering import FuseLowerer


def test_generate_gradients_emits_generate_node():
    src = """
    node m(x: f32) -> f32 {
      y = x
      loss = Add(y, 1.0)
      return loss
    }
    @train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)

    assert any(n.op_type == "GenerateGradients" and getattr(n, "domain", "") == "ai.onnx.preview.training" for n in model.graph.node)
    assert any(o.name.endswith("W.grad") for o in model.graph.output)
