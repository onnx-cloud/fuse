import pytest
pytest.importorskip("lark")
pytest.importorskip("onnx")
from src.parser import fuse_parser
from src.lowering import FuseLowerer


def test_dynamic_lr_and_step_exposed_in_algorithm_inputs():
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

    # algorithm inputs should include lr and step (strip-scope names)
    inp_names = [i.name for i in alg.input]
    assert any("lr" in n for n in inp_names), f"no lr input found in {inp_names}"
    assert any("step" in n for n in inp_names), f"no step input found in {inp_names}"
