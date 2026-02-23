from src.lowering import FuseLowerer
from src.lowering.onnx_emitter import InMemoryONNXEmitter
from src.name_allocator import StableNameAllocator
from src.parser import fuse_parser
from tests.test_utils import project_fuse_version
FUSE_DECL = f"@fuse {project_fuse_version()}\n"


def test_weights_become_initializers():
    src = FUSE_DECL + """
    weight W: f32[2,2] = [[1.0, 0.0], [0.0, 1.0]]
    node m(x: f32[2]) -> f32[2] { return MatMul(W, x) }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer()
    model = fl.lower(ast)
    assert model is not None
    inits = [i.name for i in model.graph.initializer]
    # Accept qualified or unqualified names; require that `W` appears in some initializer name
    assert any(
        name.endswith("W") or name.endswith(".W") for name in inits
    ), f"Expected initializer for weight 'W', got: {inits}"


def test_export_is_deterministic_bytes():
    src = """
    const C: f32 = 1.0
    node m(x: f32[2]) -> f32[2] {
        y = Add(x, C)
        return y
    }
    """
    ast = fuse_parser.parse(src)
    fl1 = FuseLowerer()
    m1 = fl1.lower(ast, name_allocator=StableNameAllocator())
    data1 = InMemoryONNXEmitter().save_model_bytes(m1)

    # Lower again with a fresh lowerer and allocator to ensure repeatability
    fl2 = FuseLowerer()
    m2 = fl2.lower(ast, name_allocator=StableNameAllocator())
    data2 = InMemoryONNXEmitter().save_model_bytes(m2)

    assert data1 == data2, "Exported ONNX bytes should be deterministic"
