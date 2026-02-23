from src.parser import fuse_parser
from src.lowering import FuseLowerer
from src.lowering.training_checks import validate_training_info
from tests.test_utils import project_fuse_version
FUSE_DECL = f"@fuse {project_fuse_version()}\n"


def test_emit_training_info_creates_traininginfo():
    src = FUSE_DECL + """
    @train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]
    @training { optimizer = Adam, lr = 1e-3 }
    node m(x: f32[2]) -> f32[2] { return MatMul(W, x) }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)
    # Ensure we emitted a TrainingInfoProto entry and it validates
    assert len(model.training_info) >= 1
    validate_training_info(model)  # should not raise


def test_grad_output_passes_through_producer():
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

    # Output should include the canonical gradient name
    outs = [o.name for o in model.graph.output]
    assert any(name.endswith("W.grad") for name in outs)

    # There should NOT be a synthetic Identity node whose output is W.grad
    identity_produces_grad = any(
        (n.op_type == "Identity" and any(o.endswith("W.grad") for o in n.output))
        for n in model.graph.node
    )
    assert not identity_produces_grad
