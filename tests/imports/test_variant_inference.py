"""Tests for variant inference from ONNX models."""

import pytest
import onnx
from onnx import TensorProto, helper

from src.imports.variant_inference import (
    infer_variant_from_elem_type,
    infer_variant_from_model,
    infer_variant_from_model_metadata,
)


class TestInferVariantFromElemType:
    """Test mapping of ONNX elem_types to variant names."""

    def test_int8_inference(self):
        """int8 should map to 'int8'."""
        assert infer_variant_from_elem_type(TensorProto.INT8) == "int8"

    def test_uint8_inference(self):
        """CRITICAL: uint8 should map to 'u8' (not 'int8')."""
        # This test addresses issue #287
        assert infer_variant_from_elem_type(TensorProto.UINT8) == "u8"

    def test_uint8_vs_int8_distinction(self):
        """Verify that UINT8 and INT8 are distinguishable."""
        int8_variant = infer_variant_from_elem_type(TensorProto.INT8)
        uint8_variant = infer_variant_from_elem_type(TensorProto.UINT8)
        assert int8_variant != uint8_variant, "INT8 and UINT8 must produce different variants"
        assert int8_variant == "int8"
        assert uint8_variant == "u8"

    def test_float32_inference(self):
        """float32 should map to 'fp32'."""
        assert infer_variant_from_elem_type(TensorProto.FLOAT) == "fp32"

    def test_float16_inference(self):
        """float16 should map to 'f16'."""
        assert infer_variant_from_elem_type(TensorProto.FLOAT16) == "f16"

    def test_int64_inference(self):
        """int64 should map to 'int64'."""
        assert infer_variant_from_elem_type(TensorProto.INT64) == "int64"

    def test_uint16_inference(self):
        """uint16 should map to 'u16'."""
        assert infer_variant_from_elem_type(TensorProto.UINT16) == "u16"

    def test_bool_inference(self):
        """bool should map to 'bool'."""
        assert infer_variant_from_elem_type(TensorProto.BOOL) == "bool"

    def test_unsupported_type_raises(self):
        """Unknown elem_type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown or unsupported"):
            infer_variant_from_elem_type(999)  # Invalid type code


class TestInferVariantFromModel:
    """Test variant inference from ONNX model initializers."""

    def test_single_float_initializer(self):
        """Model with all float initializers should infer 'fp32'."""
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Add", ["x", "w"], ["y"]),
                ],
                "test",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
                initializer=[
                    helper.make_tensor(
                        "w", TensorProto.FLOAT, [3], [1.0, 2.0, 3.0]
                    ),
                ],
            ),
        )
        assert infer_variant_from_model(model) == "fp32"

    def test_single_int8_initializer(self):
        """Model with all int8 initializers should infer 'int8'."""
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Add", ["x", "w"], ["y"]),
                ],
                "test",
                [helper.make_tensor_value_info("x", TensorProto.INT8, [1, 3])],
                [helper.make_tensor_value_info("y", TensorProto.INT8, [1, 3])],
                initializer=[
                    helper.make_tensor(
                        "w", TensorProto.INT8, [3], [1, 2, 3], raw=False
                    ),
                ],
            ),
        )
        assert infer_variant_from_model(model) == "int8"

    def test_single_uint8_initializer(self):
        """Model with all uint8 initializers should infer 'u8'."""
        # This is the critical test for issue #287
        import numpy as np

        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Add", ["x", "w"], ["y"]),
                ],
                "test",
                [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 3])],
                [helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 3])],
                initializer=[
                    helper.make_tensor(
                        "w",
                        TensorProto.UINT8,
                        [3],
                        np.array([1, 2, 3], dtype=np.uint8).tobytes(),
                        raw=True,
                    ),
                ],
            ),
        )
        variant = infer_variant_from_model(model)
        assert variant == "u8", f"Expected 'u8' for UINT8 initializer, got '{variant}'"

    def test_mixed_dtypes_returns_none(self):
        """Model with mixed-dtype initializers should return None."""
        import numpy as np

        graph = helper.make_graph(
            [
                helper.make_node("Add", ["x", "w1"], ["temp"]),
                helper.make_node("Add", ["temp", "w2"], ["y"]),
            ],
            "test",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
            initializer=[
                helper.make_tensor("w1", TensorProto.FLOAT, [3], [1.0, 2.0, 3.0]),
                helper.make_tensor(
                    "w2",
                    TensorProto.INT8,
                    [3],
                    np.array([1, 2, 3], dtype=np.int8).tobytes(),
                    raw=True,
                ),
            ],
        )
        model = helper.make_model(graph)
        assert infer_variant_from_model(model) is None

    def test_no_initializers_returns_none(self):
        """Model with no initializers should return None."""
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Add", ["x", "y"], ["z"]),
                ],
                "test",
                [
                    helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3]),
                    helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3]),
                ],
                [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])],
            ),
        )
        assert infer_variant_from_model(model) is None


class TestInferVariantFromModelMetadata:
    """Test extracting variant from model metadata."""

    def test_variant_in_metadata(self):
        """If 'variant' key exists in metadata_props, return its value."""
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["x"], ["y"])],
                "test",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
            ),
        )
        # Add metadata using StringStringEntryProto
        from onnx import StringStringEntryProto
        entry = StringStringEntryProto()
        entry.key = "variant"
        entry.value = "fp32_optimized"
        model.metadata_props.append(entry)

        assert infer_variant_from_model_metadata(model) == "fp32_optimized"

    def test_no_variant_in_metadata(self):
        """If 'variant' key not in metadata_props, return None."""
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["x"], ["y"])],
                "test",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
            ),
        )
        # Add other metadata but not 'variant'
        from onnx import StringStringEntryProto
        entry = StringStringEntryProto()
        entry.key = "author"
        entry.value = "test"
        model.metadata_props.append(entry)

        assert infer_variant_from_model_metadata(model) is None

    def test_empty_metadata_returns_none(self):
        """Model with no metadata_props should return None."""
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["x"], ["y"])],
                "test",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
            ),
        )
        # Don't add any metadata
        assert infer_variant_from_model_metadata(model) is None


class TestRoundTripUint8Variant:
    """Integration test: create UINT8 model, infer variant, verify round-trip."""

    def test_uint8_round_trip(self):
        """Create UINT8 model, infer variant, verify it's 'u8' and distinct from INT8."""
        import numpy as np

        # Create a uint8 model
        uint8_model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Add", ["x", "w"], ["y"])],
                "test_uint8",
                [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 3])],
                [helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 3])],
                initializer=[
                    helper.make_tensor(
                        "w",
                        TensorProto.UINT8,
                        [3],
                        np.array([10, 20, 30], dtype=np.uint8).tobytes(),
                        raw=True,
                    ),
                ],
            ),
        )

        # Infer variant from model
        variant = infer_variant_from_model(uint8_model)
        assert variant == "u8"

        # Verify it's different from INT8 variant
        int8_model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Add", ["x", "w"], ["y"])],
                "test_int8",
                [helper.make_tensor_value_info("x", TensorProto.INT8, [1, 3])],
                [helper.make_tensor_value_info("y", TensorProto.INT8, [1, 3])],
                initializer=[
                    helper.make_tensor(
                        "w",
                        TensorProto.INT8,
                        [3],
                        np.array([10, 20, 30], dtype=np.int8).tobytes(),
                        raw=True,
                    ),
                ],
            ),
        )

        int8_variant = infer_variant_from_model(int8_model)
        assert int8_variant == "int8"
        assert variant != int8_variant
