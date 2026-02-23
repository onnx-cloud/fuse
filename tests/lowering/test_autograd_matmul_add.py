from src.parser import fuse_parser
from src.lowering import FuseLowerer


def test_autograd_wrt_weight():
    src = """
    @train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]
    node m(x: f32[2]) -> f32 {
      w = MatMul(W, x)
      loss = Add(w, 1.0)
      return loss
    }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)
    # Expect a gradient output for the parameter
    assert any(o.name.endswith("W.grad") for o in model.graph.output)
    # And expect MatMul/Transpose nodes in training domain for the gradient
    assert any(n.op_type in ("MatMul", "Transpose") and getattr(n, "domain", "") == "ai.onnx.preview.training" for n in model.graph.node)
