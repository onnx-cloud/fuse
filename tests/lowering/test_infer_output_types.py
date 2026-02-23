import onnx
from onnx import TensorProto
from src.lowering import FuseLowerer
from src.parser import fuse_parser


def test_infer_comparison_output_type():
    from importlib.metadata import version as _pkg_version
    FUSE_DECL = f"@fuse {_pkg_version('fuse')}\n"
    src = FUSE_DECL + """
    node cmp(x: f32[2], y: f32[2]) {
      return Greater(x, y)
    }
    """
    ast = fuse_parser.parse(src)
    model = FuseLowerer().lower(ast)
    onnx.checker.check_model(model)
    out = model.graph.output[0]
    assert getattr(out.type.tensor_type, "elem_type") == int(TensorProto.BOOL)


def test_infer_elementwise_output_type():
    from importlib.metadata import version as _pkg_version
    FUSE_DECL = f"@fuse {_pkg_version('fuse')}\n"
    src = FUSE_DECL + """
    node add(x: f32[2], y: f32[2]) {
      return x + y
    }
    """
    ast = fuse_parser.parse(src)
    model = FuseLowerer().lower(ast)
    onnx.checker.check_model(model)
    out = model.graph.output[0]
    assert getattr(out.type.tensor_type, "elem_type") == int(TensorProto.FLOAT)


def test_infer_cast_to_output_type():
    from importlib.metadata import version as _pkg_version
    FUSE_DECL = f"@fuse {_pkg_version('fuse')}\n"
    src = FUSE_DECL + """
    node c(x: i64) {
      return Cast<f32>(x)
    }
    """
    ast = fuse_parser.parse(src)
    model = FuseLowerer().lower(ast)
    onnx.checker.check_model(model)
    out = model.graph.output[0]
    assert getattr(out.type.tensor_type, "elem_type") == int(TensorProto.FLOAT)
