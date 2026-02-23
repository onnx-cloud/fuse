import pytest
pytest.importorskip("lark")
pytest.importorskip("onnx")
from src.parser import fuse_parser
from src.lowering import FuseLowerer


def test_explicit_algorithm_preferred_and_respects_signature():
    src = """
    @train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]
    @training { optimizer = Adama, lr = 1e-3 }

    graph my_alg(W: f32[2,2], lr: f32[1]) -> f32[2,2] {
        W_new = Identity(W)
        W_new
    }

    node m(x: f32[2]) -> f32[2] {
        y = MatMul(W, x)
        y
    }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)

    ti = model.training_info[0]
    alg = ti.algorithm

    inp_names = [i.name for i in alg.input]
    assert any('lr' in n for n in inp_names)
