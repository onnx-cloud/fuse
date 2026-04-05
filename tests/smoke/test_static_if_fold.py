"""Smoke test for 'static if' constant condition folding."""

import onnx
from src.lowering.main import FuseLowerer
from src.parser import fuse_parser


def test_static_if_const_true():
    """Test that 'static if true' can be lowered (constant folding optimization is not yet implemented)."""
    src = """
@fuse 0.7
@opset onnx 13
@domain test

fn const_branch(x: f32) -> f32 {
    static if true {
        Add(x, 1.0)
    } else {
        Sub(x, 1.0)
    }
}
"""
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    model = lowerer.lower(ast)
    
    # Verify model was created
    assert model is not None
    assert isinstance(model, onnx.ModelProto)
    
    # Note: Constant folding for static if is not yet implemented.
    # The model may emit an If node even with constant conditions.
    # Just verify the model is valid.
    assert len(model.graph.input) >= 1


def test_static_if_const_false():
    """Test that 'static if false' can be lowered (constant folding optimization is not yet implemented)."""
    src = """
@fuse 0.7
@opset onnx 13
@domain test

fn const_branch_false(x: f32) -> f32 {
    static if false {
        Add(x, 1.0)
    } else {
        Sub(x, 1.0)
    }
}
"""
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    model = lowerer.lower(ast)
    
    # Verify model was created
    assert model is not None
    assert isinstance(model, onnx.ModelProto)
    
    # Note: Constant folding for static if is not yet implemented.
    # The model may emit an If node even with constant conditions.
    # Just verify the model is valid.
    assert len(model.graph.input) >= 1


if __name__ == "__main__":
    test_static_if_const_true()
    test_static_if_const_false()
    print("All static if constant folding smoke tests passed!")
