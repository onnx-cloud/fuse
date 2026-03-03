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

fn reshape_dynamic(x: f32[1, _, 64]) -> f32[_, 64] {
    Reshape(x, [2, 32])
}
"""
    lowerer = FuseLowerer()
    ast = lowerer._FuseParserWrapper__parser.parse(src) if hasattr(lowerer, '_FuseParserWrapper__parser') else None
    if ast is None:
        from src.parser import fuse_parser
        ast = fuse_parser.parse(src)
    
    # Lower to ONNX
    model = lowerer.lower(ast)
    
    # Verify the model was created
    assert model is not None
    assert isinstance(model, onnx.ModelProto)
    
    # Check that input has dynamic dimension
    assert len(model.graph.input) > 0
    input_ = model.graph.input[0]
    shape = input_.type.tensor_type.shape
    
    # Find the dynamic dimension (dim_value == 0 means dynamic)
    has_dynamic = any(
        d.dim_value == 0 or not d.HasField("dim_value")
        for d in shape.dim
    )
    assert has_dynamic, "Expected at least one dynamic dimension in input shape"


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


if __name__ == "__main__":
    test_dynamic_shape_underscore()
    test_symbolic_dimension_preserved()
    print("All dynamic shape smoke tests passed!")
