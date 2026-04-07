"""Smoke tests for dynamic shapes with '_' and symbolic dimensions."""

import onnx
from onnx import TensorProto, helper
from src.lowering.main import FuseLowerer
from src.graph_context import GraphContext


def test_dynamic_shape_underscore():
    """Test that '_' in shape represents a dynamic dimension."""
    src = """
@fuse 0.7
@opset onnx 13
@domain test

fn reshape_dynamic(x: f32[1, _, 64]) -> f32[1, _, 64] {
    x
}
"""
    from src.parser import fuse_parser
    ast = fuse_parser.parse(src)
    
    lowerer = FuseLowerer()
    model = lowerer.lower(ast)
    
    # Verify the model was created
    assert model is not None
    assert isinstance(model, onnx.ModelProto)
    
    # Check that input has dynamic dimension
    assert len(model.graph.input) > 0, "Expected at least one input"
    input_ = model.graph.input[0]
    shape = input_.type.tensor_type.shape
    assert len(shape.dim) == 3, f"Expected 3 input dims, got {len(shape.dim)}"
    
    # Verify dimension values: [1, dynamic, 64]
    assert shape.dim[0].dim_value == 1, "First dim should be 1"
    # Second dim should be dynamic (dim_value == 0)
    assert shape.dim[1].dim_value == 0, "Second dim should be dynamic (0)"
    assert shape.dim[2].dim_value == 64, "Third dim should be 64"
    
    # Output should match input shape for passthrough
    output = model.graph.output[0]
    output_shape = output.type.tensor_type.shape
    assert len(output_shape.dim) == 3, f"Expected 3 output dims, got {len(output_shape.dim)}"
    assert output_shape.dim[0].dim_value == 1, "Output first dim should be 1"
    assert output_shape.dim[1].dim_value == 0, "Output second dim should be dynamic"
    assert output_shape.dim[2].dim_value == 64, "Output third dim should be 64"
    
    # Verify type consistency
    assert input_.type.tensor_type.elem_type == TensorProto.FLOAT, "Input should be float32"
    assert output.type.tensor_type.elem_type == TensorProto.FLOAT, "Output should be float32"


def test_symbolic_dimension_preserved():
    """Test that symbolic dimensions (e.g., batch_size) are preserved."""
    src = """
@fuse 0.7
@opset onnx 13
@domain test

fn symbolic_shape(x: f32[batch, 64], scale: f32 = 1.0) -> f32[batch, 64] {
    Mul(x, scale)
}
"""
    from src.parser import fuse_parser
    ast = fuse_parser.parse(src)
    
    lowerer = FuseLowerer()
    model = lowerer.lower(ast)
    
    # Verify model was created
    assert model is not None
    assert isinstance(model, onnx.ModelProto)
    
    # Check inputs: should have 2 (x and scale)
    assert len(model.graph.input) >= 1, "Expected at least one input"
    main_input = model.graph.input[0]
    
    # Verify input shape has 2 dims
    shape = main_input.type.tensor_type.shape
    assert len(shape.dim) == 2, f"Expected 2 input dims, got {len(shape.dim)}"
    
    # First dim should be symbolic/dynamic (batch)
    has_batch_dim = shape.dim[0].dim_value == 0 or not shape.dim[0].HasField("dim_value")
    assert has_batch_dim, "Batch dimension should be dynamic/symbolic"
    
    # Second dim should be 64
    assert shape.dim[1].dim_value == 64, "Second dim should be 64"
    
    # Check output shape matches input shape
    assert len(model.graph.output) > 0, "Expected at least one output"
    output = model.graph.output[0]
    output_shape = output.type.tensor_type.shape
    
    assert len(output_shape.dim) == 2, f"Expected 2 output dims, got {len(output_shape.dim)}"
    assert output_shape.dim[1].dim_value == 64, "Output second dim should match input (64)"
    
    # Verify graph has Mul node
    mul_nodes = [n for n in model.graph.node if n.op_type == "Mul"]
    assert len(mul_nodes) > 0, "Expected at least one Mul node"


if __name__ == "__main__":
    test_dynamic_shape_underscore()
    test_symbolic_dimension_preserved()
    print("All dynamic shape smoke tests passed!")
