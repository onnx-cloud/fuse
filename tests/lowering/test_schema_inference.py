from src.lowering.schema_inference import (
    infer_output_from_schema,
    map_type_params_from_inputs,
)
from src.onnx_schema import require_op_schema


def _to_input_types(scalar, dims=None):
    return {"scalar": scalar, "dims": list(dims or [])}


def test_map_type_params_from_add_schema():
    schema = require_op_schema("Add", 13, "")
    # Add has inputs that share a type parameter 'T'
    input_types = [
        _to_input_types("f64", [2, 3]),
        _to_input_types("f64", [2, 3]),
    ]
    m = map_type_params_from_inputs(schema, input_types)
    assert any(v["scalar"] == "f64" for v in m.values())


def test_infer_output_for_add():
    schema = require_op_schema("Add", 13, "")
    input_types = [
        _to_input_types("f32", [2, 3]),
        _to_input_types("f32", [2, 3]),
    ]
    out = infer_output_from_schema(schema, input_types)
    assert out is not None and out["scalar"] == "f32"


def test_infer_output_for_matmul():
    schema = require_op_schema("MatMul", 13, "")
    input_types = [
        _to_input_types("f16", [2, 3]),
        _to_input_types("f16", [3, 4]),
    ]
    out = infer_output_from_schema(schema, input_types)
    # MatMul binds to 'T' across inputs and output; should infer f16
    assert out is not None and out["scalar"] == "f16"


def test_infer_output_for_reducesum():
    schema = require_op_schema("ReduceSum", 13, "")
    input_types = [_to_input_types("f32", [2, 3])]
    out = infer_output_from_schema(schema, input_types)
    assert out is not None and out["scalar"] == "f32"
