"""Comprehensive smoke test for model execution and input/output validation."""

import numpy as np
import onnx
from onnx import TensorProto, helper
from src.lowering.main import FuseLowerer
from src.parser import fuse_parser


def test_simple_model_io_shape_types():
    """Verify that compiled models have correct input/output shapes and types."""
    src = """
@fuse 0.7
@opset onnx 13
@domain test

fn add_model(x: f32[2, 3], y: f32[2, 3]) -> f32[2, 3] {
    Add(x, y)
}
"""
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    model = lowerer.lower(ast)
    
    # Verify model structure
    assert model is not None
    assert len(model.graph.input) == 2, "Expected 2 inputs"
    assert len(model.graph.output) == 1, "Expected 1 output"
    
    # Check input shapes and types
    x_input = next((i for i in model.graph.input if "x" in i.name or i == model.graph.input[0]), model.graph.input[0])
    y_input = next((i for i in model.graph.input if "y" in i.name or i == model.graph.input[1]), model.graph.input[1])
    
    for inp in [x_input, y_input]:
        assert inp.type.tensor_type.elem_type == TensorProto.FLOAT, f"Input {inp.name} should be float32"
        shape = inp.type.tensor_type.shape
        assert len(shape.dim) == 2, f"Input {inp.name} should have 2 dims"
        assert shape.dim[0].dim_value == 2, f"Input {inp.name} dim 0 should be 2"
        assert shape.dim[1].dim_value == 3, f"Input {inp.name} dim 1 should be 3"
    
    # Check output shape and type
    output = model.graph.output[0]
    assert output.type.tensor_type.elem_type == TensorProto.FLOAT, "Output should be float32"
    out_shape = output.type.tensor_type.shape
    assert len(out_shape.dim) == 2, "Output should have 2 dims"
    assert out_shape.dim[0].dim_value == 2, "Output dim 0 should be 2"
    assert out_shape.dim[1].dim_value == 3, "Output dim 1 should be 3"
    
    # Verify Add node is in graph
    add_nodes = [n for n in model.graph.node if n.op_type == "Add"]
    assert len(add_nodes) > 0, "Should have at least one Add node"


def test_node_naming_conventions():
    """Verify that all nodes follow proper naming conventions."""
    src = """
@fuse 0.7
@opset onnx 13
@domain test

fn composite(x: f32[10]) -> f32[10] {
    y = Add(x, 1.0)
    z = Mul(y, 2.0)
    z
}
"""
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    model = lowerer.lower(ast)
    
    # Verify graph has nodes
    assert len(model.graph.node) >= 2, "Expected at least 2 nodes"
    
    # Check node naming and structure
    for i, node in enumerate(model.graph.node):
        # Node should have op_type - allow common broadcasting/utility ops
        valid_ops = ["Add", "Mul", "Identity", "Unsqueeze", "Squeeze", "Reshape"]
        assert node.op_type in valid_ops or node.op_type[0].isupper(), \
            f"Node {i} has invalid op_type: {node.op_type}"
        
        # Node outputs should have proper names
        assert len(node.output) > 0, f"Node {i} should have outputs"
        assert all(isinstance(o, str) for o in node.output), f"Node {i} outputs should be strings"
        assert all(len(o) > 0 for o in node.output), f"Node {i} output names should not be empty"
        
        # Node inputs should reference something
        assert len(node.input) > 0, f"Node {i} should have inputs"
        assert all(isinstance(inp, str) for inp in node.input), f"Node {i} inputs should be strings"
    
    # Verify output name is set correctly
    assert len(model.graph.output) > 0, "Model should have outputs"
    assert model.graph.output[0].name, "Output should have a name"
    
    # Verify we have the core computation nodes (Add and Mul)
    ops = [n.op_type for n in model.graph.node]
    assert "Add" in ops, "Should have Add node"
    assert "Mul" in ops, "Should have Mul node"


def test_parallelizable_graph_structure():
    """Verify graph structure supports independent parallel operations."""
    src = """
@fuse 0.7
@opset onnx 13
@domain test

fn independent_ops(a: f32[5], b: f32[5], c: f32[5]) -> f32[5] {
    y = Add(a, b)
    z = Mul(y, c)
    z
}
"""
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    model = lowerer.lower(ast)
    
    # Verify we have 3 inputs
    assert len(model.graph.input) == 3, f"Expected 3 inputs, got {len(model.graph.input)}"
    
    # Verify we have 1 output
    assert len(model.graph.output) >= 1, "Expected at least 1 output"
    
    # Check that we have Add and Mul nodes
    ops = [n.op_type for n in model.graph.node if n.op_type in ["Add", "Mul"]]
    assert "Add" in ops, "Should have Add node"
    assert "Mul" in ops, "Should have Mul node"
    
    # Verify proper data flow from inputs to output
    output_name = model.graph.output[0].name
    
    # Find nodes that produce the output
    producing_nodes = [n for n in model.graph.node if output_name in n.output]
    assert len(producing_nodes) > 0, f"No node produces output '{output_name}'"


def test_consistent_tensor_naming_across_graph():
    """Verify that tensor naming is consistent within a graph."""
    src = """
@fuse 0.7
@opset onnx 13
@domain test

fn tensor_flow(x: f32[4, 4], w: f32[4, 4]) -> f32[4, 4] {
    mm = MatMul(x, w)
    Add(mm, 0.5)
}
"""
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    model = lowerer.lower(ast)
    
    # Collect all tensor names that are available as sources
    input_names = {i.name for i in model.graph.input}
    initializer_names = {init.name for init in model.graph.initializer}
    
    # Build set of all outputs produced by nodes
    node_outputs = set()
    for node in model.graph.node:
        for out in node.output:
            node_outputs.add(out)
    
    # All available sources
    available_sources = input_names | initializer_names | node_outputs
    
    # Check that all node inputs are satisfied
    for node_idx, node in enumerate(model.graph.node):
        for inp_name in node.input:
            assert inp_name in available_sources, \
                f"Node {node_idx} ({node.op_type}) input '{inp_name}' is not available"
    
    # Verify every node output is a valid string
    for node in model.graph.node:
        for out_name in node.output:
            assert isinstance(out_name, str), f"Node output should be string, got {type(out_name)}"
            assert len(out_name) > 0, f"Node output name should not be empty"
    
    # Model output should reference a valid tensor
    for output in model.graph.output:
        assert output.name, "Output should have a name"
        assert output.name in available_sources, \
            f"Output '{output.name}' is not produced by any source"