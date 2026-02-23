"""Tests for type inference module."""

import pytest
from src.lowering.type_inference import TypeInferencer
from src.graph_context import GraphContext


class TestTypeInferencer:
    """Test TypeInferencer class."""
    
    @pytest.fixture
    def ctx(self):
        """Create a GraphContext for testing."""
        ctx = GraphContext(name="test")
        # Add some test values with types
        ctx.value_types["x"] = {"scalar": "f32", "dims": [1, 3, 224, 224]}
        ctx.value_types["y"] = {"scalar": "f32", "dims": [1, 3, 224, 224]}
        ctx.value_types["mask"] = {"scalar": "bool", "dims": [1, 3, 224, 224]}
        ctx.value_types["weights"] = {"scalar": "f32", "dims": [10, 3, 3, 3]}
        return ctx
    
    @pytest.fixture
    def inferencer(self, ctx):
        """Create a TypeInferencer for testing."""
        return TypeInferencer(ctx)
    
    def test_logical_operators_return_bool(self, inferencer):
        """Test that logical operators return bool type."""
        result = inferencer.infer_output_type("Equal", ["x", "y"])
        assert result["scalar"] == "bool"
        
        result = inferencer.infer_output_type("Greater", ["x", "y"])
        assert result["scalar"] == "bool"
        
        result = inferencer.infer_output_type("And", ["mask", "mask"])
        assert result["scalar"] == "bool"
    
    def test_elementwise_operators_preserve_type(self, inferencer):
        """Test that element-wise operators preserve input type."""
        result = inferencer.infer_output_type("Add", ["x", "y"])
        assert result["scalar"] == "f32"
        assert result["dims"] == [1, 3, 224, 224]
        
        result = inferencer.infer_output_type("Mul", ["x", "y"])
        assert result["scalar"] == "f32"
    
    def test_matmul_preserves_dtype(self, inferencer):
        """Test that MatMul preserves dtype."""
        result = inferencer.infer_output_type("MatMul", ["x", "y"])
        assert result["scalar"] == "f32"
    
    def test_reduction_operators(self, inferencer):
        """Test that reduction operators preserve dtype."""
        result = inferencer.infer_output_type("ReduceSum", ["x"])
        assert result["scalar"] == "f32"
        
        result = inferencer.infer_output_type("ReduceMean", ["x"])
        assert result["scalar"] == "f32"
    
    def test_reshape_preserves_dtype(self, inferencer):
        """Test that Reshape preserves dtype but clears dims."""
        result = inferencer.infer_output_type("Reshape", ["x"])
        assert result["scalar"] == "f32"
        assert result["dims"] == []
    
    def test_concat_preserves_dtype(self, inferencer):
        """Test that Concat preserves dtype."""
        result = inferencer.infer_output_type("Concat", ["x", "y"])
        assert result["scalar"] == "f32"
    
    def test_explicit_input_types(self, inferencer):
        """Test providing explicit input types."""
        input_types = [
            {"scalar": "f16", "dims": [1, 100]},
            {"scalar": "f16", "dims": [1, 100]}
        ]
        result = inferencer.infer_output_type("Add", ["a", "b"], input_types=input_types)
        assert result["scalar"] == "f16"
    
    def test_no_input_types_returns_default(self, inferencer):
        """Test that missing input types return default."""
        result = inferencer.infer_output_type("Unknown", [])
        assert result["scalar"] == "f32"
        assert result["dims"] == []
