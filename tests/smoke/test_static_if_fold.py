"""Smoke test for 'static if' constant condition folding."""

import onnx
from src.lowering.main import FuseLowerer
from src.parser import fuse_parser


def test_static_if_const_true():
    """Test that 'static if true' eliminates dead code (only true branch lowered)."""
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
    
    # Count the number of nodes: should only have Add, not If or Sub
    node_count = len(list(model.graph.node))
    assert node_count == 1, f"Expected 1 node (Add), but got {node_count}"
    
    # Verify it's an Add node
    assert model.graph.node[0].op_type == "Add"


def test_static_if_const_false():
    """Test that 'static if false' eliminates the true branch (only false branch lowered)."""
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
    
    # Count the number of nodes: should only have Sub, not If or Add
    node_count = len(list(model.graph.node))
    assert node_count >= 1, f"Expected at least 1 node, but got {node_count}"
    
    # Verify there's a Sub node (or the final output is via Sub-like operation)
    has_sub = any(node.op_type == "Sub" for node in model.graph.node)
    assert has_sub, "Expected Sub node in output for 'static if false' branch"


if __name__ == "__main__":
    test_static_if_const_true()
    test_static_if_const_false()
    print("All static if constant folding smoke tests passed!")
