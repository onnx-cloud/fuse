"""Type inference for ONNX operators.

This module handles type and shape inference for operators during lowering,
extracting this responsibility from the main FuseLowerer class.
"""

import logging
from typing import Any, Dict, List, Optional

import onnx
from ..graph_context import GraphContext, as_tensor_type, DTYPE_MAP, ONNX_TO_FUSE

logger = logging.getLogger(__name__)

# Reverse lookup: ONNX TensorProto element type int → set of Fuse scalar names
_ONNX_ELEM_TO_FUSE_SCALARS: Dict[int, set] = {}
for _fuse_name, _onnx_int in DTYPE_MAP.items():
    _ONNX_ELEM_TO_FUSE_SCALARS.setdefault(_onnx_int, set()).add(_fuse_name)


class TypeInferencer:
    """Handles type inference for ONNX operators during lowering."""
    
    def __init__(self, ctx: GraphContext):
        self.ctx = ctx
        self.schema_cache: Dict[str, Any] = {}
    
    def infer_output_type(
        self,
        op_type: str,
        input_names: List[str],
        input_types: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Infer output type for an operator based on inputs.
        
        Uses a two-tier strategy:
        1. Try schema-based inference via ``infer_from_schema`` which consults
           the ONNX operator registry for type constraints.
        2. Fall back to heuristic rules (first-input type propagation,
           special-case operator tables).
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

        # Try schema-based inference first
        schema_result = self.infer_from_schema(op_type, input_names)
        if schema_result is not None:
            return schema_result
        
        # Logical/comparison operators always return bool
        if op_type in {
            "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual",
            "And", "Or", "Not", "Xor", "IsNaN", "IsInf",
        }:
            dims = input_types[0].get("dims", []) if input_types else []
            return {"scalar": "bool", "dims": dims}

        # Shape/size operators always return int64
        if op_type in {"Shape", "Size", "ArgMax", "ArgMin", "NonZero"}:
            return {"scalar": "i64", "dims": []}
        
        # Cast explicitly changes type — caller must supply type_hint
        if op_type == "Cast":
            dims = input_types[0].get("dims", []) if input_types else []
            return {"scalar": input_types[0].get("scalar", "f32"), "dims": dims}
        
        first_type = input_types[0] if input_types else as_tensor_type()
        
        # Shape-changing operators: preserve dtype, dims are dynamic
        if op_type in {"Reshape", "Flatten", "Squeeze", "Unsqueeze"}:
            return {"scalar": first_type.get("scalar", "f32"), "dims": []}
        
        # Reduction operators
        if op_type in {
            "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin",
            "ReduceProd", "ReduceL1", "ReduceL2",
            "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare",
        }:
            return {"scalar": first_type.get("scalar", "f32"), "dims": first_type.get("dims", [])}
        
        # MatMul and similar: preserve dtype
        if op_type in {"MatMul", "Gemm"}:
            return {"scalar": first_type.get("scalar", "f32"), "dims": first_type.get("dims", [])}
        
        # Element-wise operators: preserve first input type
        if op_type in {
            "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log",
            "Abs", "Neg", "Ceil", "Floor", "Reciprocal",
            "Relu", "Sigmoid", "Tanh", "Softmax", "LogSoftmax",
        }:
            return first_type
        
        # Concat preserves dtype
        if op_type == "Concat":
            return first_type
        
        # Default: preserve first input type
        return first_type
    
    def infer_from_schema(
        self, op_type: str, inputs: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Infer output type using the ONNX operator schema type constraints.

        Consults the ONNX operator registry for the operator's formal type
        constraints, resolves type variables from the concrete input types
        already recorded in ``self.ctx.value_types``, and returns the inferred
        output type for the first output.  Returns ``None`` when the schema is
        unavailable or the constraints cannot be resolved.
        """
        # Fast path: skip if no inputs have known types
        if not inputs:
            return None

        cache_key = f"{op_type}@{self.ctx.opset}"
        schema = self.schema_cache.get(cache_key, _SENTINEL)
        if schema is _SENTINEL:
            try:
                schema = onnx.defs.get_schema(op_type, self.ctx.opset, "")
                self.schema_cache[cache_key] = schema
            except Exception:
                self.schema_cache[cache_key] = None
                return None
        if schema is None:
            return None

        # Build a mapping from type-variable name → concrete Fuse scalar, by
        # examining which type constraint each formal input belongs to and
        # matching it against the concrete type of each actual input.
        type_var_map: Dict[str, str] = {}  # constraint name → fuse scalar

        try:
            formal_inputs = list(schema.inputs)
        except Exception:
            return None

        for idx, formal in enumerate(formal_inputs):
            if idx >= len(inputs):
                break
            actual_name = inputs[idx]
            actual_type = self.ctx.value_types.get(actual_name)
            if not actual_type:
                continue
            actual_scalar = actual_type.get("scalar")
            if not actual_scalar:
                continue
            # ``formal.typeStr`` is e.g. "T" or "T1"; correlate with constraints
            tvar = getattr(formal, "typeStr", None)
            if tvar and tvar not in type_var_map:
                type_var_map[tvar] = actual_scalar

        if not type_var_map:
            return None

        # Resolve the first output's type variable to a concrete type.
        try:
            formal_outputs = list(schema.outputs)
        except Exception:
            return None

        if not formal_outputs:
            return None

        out_tvar = getattr(formal_outputs[0], "typeStr", None)
        if out_tvar and out_tvar in type_var_map:
            resolved_scalar = type_var_map[out_tvar]
            # Gather dims from the first input as a reasonable default
            first_input_type = self.ctx.value_types.get(inputs[0]) if inputs else None
            dims = (first_input_type or {}).get("dims", [])
            return {"scalar": resolved_scalar, "dims": dims}

        # Output uses a different type variable not bound by any input we have
        # (e.g. Cast's output depends on an attribute, not an input type var).
        return None


# Sentinel for schema cache misses vs. explicit None
_SENTINEL = object()
