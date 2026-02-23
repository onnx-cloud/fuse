import pytest
pytest.importorskip("lark")
pytest.importorskip("onnx")
from src.parser import fuse_parser
from src.lowering import FuseLowerer


def test_training_config_loss_reference_creates_loss_binding():
    src = """
    @train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]
    @training { algorithm = my_alg, loss = loss_fn }

    @loss
    node loss_fn(x: f32[2]) -> f32 {
        # Fake loss
        return x[0]
    }

    @algorithm
    node my_alg(W: f32[2,2], W_grad: f32[2,2]) -> f32 {
        # produce a 'loss' output to bind
        loss = Identity(W)
        loss
    }

    node m(x: f32[2]) -> f32[2] {
        y = MatMul(W, x)
        y
    }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)

    assert len(model.training_info) >= 1
    ti = model.training_info[0]
    # loss_binding should map 'loss_fn' (training loss ref) to an algorithm output
    bindings = {e.key: e.value for e in ti.loss_binding}
    assert 'loss_fn' in bindings or 'loss' in bindings
