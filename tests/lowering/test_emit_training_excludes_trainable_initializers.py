import pytest
pytest.importorskip("lark")
pytest.importorskip("onnx")
from src.parser import fuse_parser
from src.lowering import FuseLowerer


def test_emit_training_excludes_trainable_initializers():
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

    assert len(model.training_info) >= 1
    ti = model.training_info[0]
    alg = ti.algorithm

    # Algorithm initializers should NOT contain the model weight 'W'
    init_names = [i.name for i in alg.initializer]
    assert not any(n == "W" or n.endswith(".W") or n.endswith("_W") for n in init_names)

    # update_binding should contain an entry mapping the model weight to an optimizer output
    found = any((e.key == "W" or e.key.endswith(".W") or e.key.endswith("_W")) for e in ti.update_binding)
    assert found
