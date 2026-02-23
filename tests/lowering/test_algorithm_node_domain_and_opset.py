import pytest
pytest.importorskip("lark")
pytest.importorskip("onnx")
from src.parser import fuse_parser
from src.lowering import FuseLowerer
from src.lowering.training_info_emit import TRAINING_DOMAIN


def test_algorithm_node_domain_and_opset():
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

    # Every node in algorithm should have the training domain
    assert all((getattr(n, 'domain', None) == TRAINING_DOMAIN) for n in alg.node)

    # Opset imports should include core ("" domain) and the training domain.
    # If the GraphProto does not expose opset_import, fall back to checking
    # the ModelProto.opset_import which is emitted by `GraphContext.build_model()`.
    if hasattr(alg, "opset_import"):
        domains = {o.domain for o in alg.opset_import}
    else:
        domains = {o.domain for o in model.opset_import}
    assert "" in domains
    assert TRAINING_DOMAIN in domains
