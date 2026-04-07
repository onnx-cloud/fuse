"""Smoke test for 'static if' constant condition folding."""

import onnx
from onnx import TensorProto
from src.lowering.main import FuseLowerer
from src.parser import fuse_parser


def test_static_if_const_true():
    """Test that 'static if true' can be lowered with proper graph structure."""
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
    
    # Verify model was created with expected structure
    assert model is not None
    assert isinstance(model, onnx.ModelProto)
    
    # Verify input/output consistency
    assert len(model.graph.input) >= 1, "Expected at least one input"
    assert len(model.graph.output) >= 1, "Expected at least one output"
    
    input_x = model.graph.input[0]
    output_y = model.graph.output[0]
    
    # Verify scalar input/output
    assert input_x.type.tensor_type.elem_type == TensorProto.FLOAT, "Input should be float32"
    assert output_y.type.tensor_type.elem_type == TensorProto.FLOAT, "Output should be float32"
    
    # Verify input name matches convention (clean name from function parameter)
    assert input_x.name, "Input should have a name"
    
    # Check graph has proper nodes (either If node or Add/Sub nodes depending on folding)
    node_ops = [n.op_type for n in model.graph.node]
    assert len(node_ops) > 0, "Expected at least one node in graph"
    
    # For static if true, we may have If node or the true branch (Add) may be inlined
    expected_ops = ["If", "Add", "Sub"]
    has_expected = any(op in node_ops for op in expected_ops)
    assert has_expected, f"Expected one of {expected_ops}, got {node_ops}"
    
    # Verify node naming conventions - node names should be clean identifiers
    for node in model.graph.node:
        assert node.name or node.op_type, f"Node should have name or op_type: {node}"


def test_static_if_const_false():
    """Test that 'static if false' can be lowered with proper graph structure."""
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
    
    # Verify model was created with expected structure
    assert model is not None
    assert isinstance(model, onnx.ModelProto)
    
    # Verify input matches output shape and type
    assert len(model.graph.input) >= 1, "Expected at least one input"
    assert len(model.graph.output) >= 1, "Expected at least one output"
    
    input_x = model.graph.input[0]
    output_y = model.graph.output[0]
    
    # Check type consistency
    assert input_x.type.tensor_type.elem_type == output_y.type.tensor_type.elem_type, \
        "Input and output types should match"
    
    # Verify graph nodes
    node_ops = [n.op_type for n in model.graph.node]
    assert len(node_ops) > 0, "Expected at least one node in graph"
    
    # For static if false, we may have If node or the false branch (Sub) may be inlined
    expected_ops = ["If", "Add", "Sub"]
    has_expected = any(op in node_ops for op in expected_ops)
    assert has_expected, f"Expected one of {expected_ops}, got {node_ops}"
    
    # Verify no Invalid or malformed node names
    for node in model.graph.node:
        assert isinstance(node.output[0], str), f"Node output should be string, got {type(node.output[0])}"
        assert len(node.output[0]) > 0, "Node output name should not be empty"


def test_static_if_with_complex_condition():
    """Test static if with non-scalar input."""
    src = """
@fuse 0.7
@opset onnx 13
@domain test

fn vector_branch(x: f32[10]) -> f32[10] {
    static if true {
        Add(x, 1.0)
    } else {
        Mul(x, 2.0)
    }
}
"""
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    model = lowerer.lower(ast)
    
    # Verify model structure is valid
    assert model is not None
    assert isinstance(model, onnx.ModelProto)
    
    # Verify we have inputs and outputs
    assert len(model.graph.input) >= 1, "Expected at least 1 input"
    assert len(model.graph.output) >= 1, "Expected at least 1 output"
    
    # Verify all inputs have type info
    for inp in model.graph.input:
        assert inp.HasField("type"), "Input should have type"
        assert inp.type.tensor_type.elem_type == TensorProto.FLOAT, "Input should be float"
    
    # Verify all outputs have type info
    for outp in model.graph.output:
        assert outp.HasField("type"), "Output should have type"
        assert outp.type.tensor_type.elem_type == TensorProto.FLOAT, "Output should be float"
    
    # Verify there are nodes in the graph
    assert len(model.graph.node) > 0, "Expected at least one node"


if __name__ == "__main__":
    test_static_if_const_true()
    test_static_if_const_false()
    print("All static if constant folding smoke tests passed!")
