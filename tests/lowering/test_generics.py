import pytest

from src import parser as fuse_parser
from src.lowering import FuseLowerer
from src.lowering.utils import LoweringError


def test_type_alias_shorthand_and_node_lower():
    src = """
    type T f32[N, features]

    param W0: f32[features, 32]
    param b0: f32[32]

    node dense_relu(x: T) -> f32[N, 32] {
      m = MatMul(x, W0)
      y = Add(m, b0)
      Relu(y)
    }
    """
    ast = fuse_parser.fuse_parser.parse(src)
    fl = FuseLowerer()
    # Should lower without raising
    model = fl.lower(ast)
    assert model is not None

    # Verify emitted ONNX shapes: input `x` should have dims [N, features]

    def _matches_name(n, base):
        if n == base:
            return True
        return n.endswith(f".{base}") or n.endswith(f"_{base}")

    # find input for x
    x_vi = None
    for vi in model.graph.input:
        if _matches_name(vi.name, "x"):
            x_vi = vi
            break
    assert x_vi is not None
    x_shape = x_vi.type.tensor_type.shape.dim
    assert x_shape[0].dim_param == "N"
    assert x_shape[1].dim_param == "features"

    # find input for W0 and check shape [features, 32]
    w_vi = None
    for vi in model.graph.input:
        if _matches_name(vi.name, "W0"):
            w_vi = vi
            break
    assert w_vi is not None
    w_shape = w_vi.type.tensor_type.shape.dim
    assert w_shape[0].dim_param == "features"
    # second dim should be concrete 32
    assert w_shape[1].dim_value == 32


def test_type_alias_equals_form_and_node_lower():
    src = """
    type T = f32[N, features]

    param W0: f32[features, 32]
    param b0: f32[32]

    node dense_relu(x: T) -> f32[N, 32] {
      m = MatMul(x, W0)
      y = Add(m, b0)
      Relu(y)
    }
    """
    ast = fuse_parser.fuse_parser.parse(src)
    fl = FuseLowerer()
    model = fl.lower(ast)
    assert model is not None


def test_matmul_dim_mismatch_raises():
    # Wrong weight shape (features+1 mismatch) -> should raise informative error
    src = """
    type T f32[N, features]

    param W0: f32[features_plus_one, 32]
    param b0: f32[32]

    node dense_relu(x: T) -> f32[N, 32] {
      m = MatMul(x, W0)
      y = Add(m, b0)
      Relu(y)
    }
    """
    ast = fuse_parser.fuse_parser.parse(src)
    fl = FuseLowerer()
    with pytest.raises(LoweringError) as exc:
        fl.lower(ast)
    assert "MatMul dimension mismatch" in str(exc.value)
