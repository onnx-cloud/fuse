"""Type inference for ONNX operators.

This module handles type and shape inference for operators during lowering,
extracting this responsibility from the main FuseLowerer class.
"""

from typing import Any, Dict, List, Optional

from ..graph_context import GraphContext, as_tensor_type
from ..errors import E011_TypeInferenceFailed


class TypeInferencer:
    """Handles type inference for ONNX operators during lowering."""
    
    def __init__(self, ctx: GraphContext):
        """Initialize type inferencer.
        
        Args:
            ctx: GraphContext containing type information for values
        """
        self.ctx = ctx
        self.schema_cache: Dict[str, Any] = {}
    
    def infer_output_type(
        self,
        op_type: str,
        input_names: List[str],
        input_types: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Infer output type for an operator based on inputs.
        
        Args:
            op_type: ONNX operator type (e.g., "Add", "MatMul")
            input_names: List of input value names
            input_types: Optional list of input type dicts
            
        Returns:
            Type dict with 'scalar' and 'dims' keys
            
        Raises:
            E011_TypeInferenceFailed: If type cannot be inferred
        """
        # Get input types from context if not provided
        if input_types is None:
            input_types = []
            for name in input_names:
                t = self.ctx.value_types.get(name)
                if t:
                    input_types.append(t)
        
        # Short circuit: no input types available
        if not input_types:
            return as_tensor_type()
        
        # Logical/comparison operators always return bool
        if op_type in {"Equal", "Greater", "Less", "And", "Or", "Not", "Xor"}:
            dims = input_types[0].get("dims", []) if input_types else []
            return {"scalar": "bool", "dims": dims}
        
        # Cast explicitly changes type
        if op_type == "Cast":
            # For Cast, the 'to' attribute determines output type
            # This should be handled by caller passing type_hint
            dims = input_types[0].get("dims", []) if input_types else []
            return {"scalar": input_types[0].get("scalar", "f32"), "dims": dims}
        
        # Most operators preserve input type
        # For operators with multiple inputs, use first input's type
        first_type = input_types[0] if input_types else as_tensor_type()
        
        # Handle shape-changing operators
        if op_type in {"Reshape", "Flatten", "Squeeze", "Unsqueeze"}:
            # Preserve dtype but dims are dynamic
            return {"scalar": first_type.get("scalar", "f32"), "dims": []}
        
        # Reduction operators: reduce dimensions
        if op_type in {"ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"}:
            # Keep dtype, dims depend on keepdims attribute (handled elsewhere)
            return {"scalar": first_type.get("scalar", "f32"), "dims": first_type.get("dims", [])}
        
        # MatMul and similar: preserve dtype
        if op_type in {"MatMul", "Gemm"}:
            return {"scalar": first_type.get("scalar", "f32"), "dims": first_type.get("dims", [])}
        
        # Element-wise operators: preserve first input type
        if op_type in {"Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log"}:
            return first_type
        
        # Concat preserves dtype
        if op_type == "Concat":
            return first_type
        
        # Default: preserve first input type
        return first_type
    
    def infer_from_schema(self, op_type: str, inputs: List[str]) -> Optional[Dict[str, Any]]:
        """Attempt to infer type using ONNX schema (future enhancement).
        
        Args:
            op_type: Operator type
            inputs: Input names
            
        Returns:
            Inferred type or None if schema not available
        """
        # TODO: Implement schema-driven type constraint inference
        # This would parse ONNX schema type_constraints to map type variables
        # to concrete types based on input types
        return None
