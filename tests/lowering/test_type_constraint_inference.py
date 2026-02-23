from src.lowering import FuseLowerer
from src.lowering.utils import _onnx_to_fuse_scalar
from src.parser import fuse_parser


def _lower_and_get_scalar(src: str) -> str:
    ast = fuse_parser.parse(src)
    fl = FuseLowerer()
    model = fl.lower(ast)
    out = model.graph.output[0]
    elem_type = out.type.tensor_type.elem_type
    return _onnx_to_fuse_scalar(elem_type)


def test_concat_infers_output_scalar():
    src = """
    node m(a: f32[2,3], b: f32[2,3]) -> f32[2,3] {
        y = Concat(a, b, axis=1)
        return y
    }
    """
    assert _lower_and_get_scalar(src) == "f32"


def test_add_infers_output_scalar():
    src = """
    node m(a: f64[2,3], b: f64[2,3]) -> f64[2,3] {
        y = Add(a, b)
        return y
    }
    """
    assert _lower_and_get_scalar(src) == "f64"


def test_matmul_infers_output_scalar():
    src = """
    node m(a: f16[2,3], b: f16[3,4]) -> f16[2,4] {
        y = MatMul(a, b)
        return y
    }
    """
    assert _lower_and_get_scalar(src) == "f16"


def test_reducesum_infers_output_scalar():
    src = """
    node m(a: f32[2,3]) -> f32[2,3] {
        y = ReduceSum(a, axes=[1])
        return y
    }
    """
    assert _lower_and_get_scalar(src) == "f32"
