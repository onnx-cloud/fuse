from src.parser import fuse_parser
from src.lowering import FuseLowerer


def _opset_for(model, domain):
    for o in model.opset_import:
        if o.domain == domain:
            return int(o.version)
    return None


def test_optimizer_node_emitted_and_opset_added():
    src = """
    @training { optimizer = adam, lr = 0.01 }
    @train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]
    node m(x: f32[2]) -> f32[2] { return MatMul(W, x) }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)

    # Adam optimizer node should be present in the training domain
    assert any(n.op_type == "Adam" and getattr(n, "domain", "") == "ai.onnx.preview.training" for n in model.graph.node)

    # The training opset should be present (default 1)
    v = _opset_for(model, "ai.onnx.preview.training")
    assert v is not None and v >= 1


def test_respects_explicit_training_opset():
    src = """
    @opset ai.onnx.preview.training 2
    @training { optimizer = adagrad }
    @train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]
    node m(x: f32[2]) -> f32[2] { return MatMul(W, x) }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)

    v = _opset_for(model, "ai.onnx.preview.training")
    assert v == 2
    assert any(n.op_type == "Adagrad" and getattr(n, "domain", "") == "ai.onnx.preview.training" for n in model.graph.node)
