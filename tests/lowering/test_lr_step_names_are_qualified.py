import pytest
pytest.importorskip("lark")
pytest.importorskip("onnx")
from src.parser import fuse_parser
from src.lowering import FuseLowerer


def test_lr_step_names_are_qualified():
    src = """
    @train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]
    @training { optimizer = Adam, lr = 1e-3, lr_input = true, step_input = true }
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
    alg = ti.algorithm

    # inputs should include qualified lr and step names that start with a scope
    inp_names = [i.name for i in alg.input]
    assert any(n.endswith('.lr') or n.endswith('.step') for n in inp_names), f"expected qualified names in {inp_names}"
