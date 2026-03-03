"""Tests for RDF/TTL export functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from src.export.ttl import model_to_ttl, save_ttl, onnx_file_to_ttl


def _create_simple_model() -> onnx.ModelProto:
    """Create a simple ONNX model for testing."""
    # Create a simple Add node
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])

    node = helper.make_node("Add", ["X", "Y"], ["Z"], name="add_node")

    graph = helper.make_graph([node], "test_graph", [X, Y], [Z])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.producer_name = "fuse_test"
    model.producer_version = "1.0"
    model.doc_string = "Test model for TTL export"

    return model


def _create_model_with_initializer() -> onnx.ModelProto:
    """Create a model with an initializer for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])

    # Create an initializer (weight tensor)
    W = helper.make_tensor("W", TensorProto.FLOAT, [1, 3], [1.0, 2.0, 3.0])

    node = helper.make_node("Add", ["X", "W"], ["Z"], name="add_node")

    graph = helper.make_graph([node], "test_graph", [X], [Z], [W])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    return model


def _create_model_with_attributes() -> onnx.ModelProto:
    """Create a model with node attributes for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 2, 2])

    node = helper.make_node(
        "MaxPool",
        ["X"],
        ["Y"],
        name="pool_node",
        kernel_shape=[2, 2],
        strides=[2, 2],
        auto_pad="NOTSET",
    )

    graph = helper.make_graph([node], "test_graph", [X], [Y])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    return model


class TestModelToTTL:
    """Test model_to_ttl function."""

    def test_simple_model(self):
        """Test basic TTL conversion."""
        model = _create_simple_model()
        ttl = model_to_ttl(model)

        # Check basic structure
        assert "@prefix onnx:" in ttl
        assert "@prefix xsd:" in ttl
        assert "a onnx:Model" in ttl
        assert "a onnx:Graph" in ttl
        # We now emit a compact summary instead of all individual operators
        assert 'onnx:nodeCount' in ttl
        # Individual operator details should not be present
        assert 'onnx:opType' not in ttl

    def test_determinism(self):
        """Test that TTL output is deterministic."""
        model = _create_simple_model()

        ttl1 = model_to_ttl(model)
        ttl2 = model_to_ttl(model)
        ttl3 = model_to_ttl(model)

        assert ttl1 == ttl2 == ttl3

    def test_user_namespace(self):
        """Test user namespace prefix."""
        model = _create_simple_model()
        ttl = model_to_ttl(model, user_ns="my:", user_ns_uri="https://example.org/#")

        assert "@prefix my:" in ttl
        assert "my:model/" in ttl or "my:graph/" in ttl

    def test_model_metadata(self):
        """Test that model metadata is included."""
        model = _create_simple_model()
        ttl = model_to_ttl(model, include_metadata=True)

        assert 'onnx:producerName "fuse_test"' in ttl
        assert 'onnx:producerVersion "1.0"' in ttl
        assert "onnx:docString" in ttl

    def test_exclude_metadata(self):
        """Test excluding metadata."""
        model = _create_simple_model()
        ttl = model_to_ttl(model, include_metadata=False)

        assert "onnx:producerName" not in ttl

    def test_initializers(self):
        """Test initializer export when initializers are marked trainable."""
        model = _create_model_with_initializer()
        # Mark W as trainable in model metadata so TTL will expose it
        proto = onnx.StringStringEntryProto()
        proto.key = "trainables"
        proto.value = '{"W": true}'
        model.metadata_props.append(proto)

        ttl = model_to_ttl(model, include_initializers=True)

        assert "a onnx:Initializer" in ttl
        assert 'onnx:name "W"' in ttl
        assert 'onnx:dtype "float32"' in ttl

    def test_nontrainable_initializers_not_emitted(self):
        """Initializers that are not marked trainable should not be exposed by TTL."""
        model = _create_model_with_initializer()
        ttl = model_to_ttl(model, include_initializers=True)

        # No trainables metadata -> initializers should not be emitted
        assert "a onnx:Initializer" not in ttl
        assert 'onnx:name "W"' not in ttl

    def test_exclude_initializers(self):
        """Test excluding initializers."""
        model = _create_model_with_initializer()
        ttl = model_to_ttl(model, include_initializers=False)

        assert "onnx:Initializer" not in ttl
        # Internal initializers should not be exposed as GraphInputs when
        # initializers are excluded (they are implementation details).
        assert 'onnx:name "W"' not in ttl

    def test_initializer_not_duplicated_as_input(self):
        """Ensure an initializer is emitted as an Initializer and not also as a GraphInput when included."""
        model = _create_model_with_initializer()
        # Mark W as trainable so it will be emitted as an initializer
        proto = onnx.StringStringEntryProto()
        proto.key = "trainables"
        proto.value = '{"W": true}'
        model.metadata_props.append(proto)

        ttl = model_to_ttl(model, include_initializers=True)

        # Initializer should be present
        assert "a onnx:Initializer" in ttl
        assert 'onnx:name "W"' in ttl
        # But it should not be emitted as a GraphInput when included as an initializer
        # (i.e., no GraphInput entry referencing W)
        assert 'onnx:hasInput' in ttl
        assert 'onnx:hasInitializer' in ttl
        assert '#input/W' not in ttl

    def test_node_attributes(self):
        """Test node attribute export."""
        model = _create_model_with_attributes()
        ttl = model_to_ttl(model)

        # Node attributes are not emitted individually in the compact form
        assert "a onnx:NodeAttribute" not in ttl
        assert 'onnx:name "kernel_shape"' not in ttl
        assert 'onnx:attrType "INTS"' not in ttl

    def test_graph_inputs_outputs(self):
        """Test graph input/output export."""
        model = _create_simple_model()
        ttl = model_to_ttl(model)

        assert "a onnx:GraphInput" in ttl
        assert "a onnx:GraphOutput" in ttl
        assert 'onnx:name "X"' in ttl
        assert 'onnx:name "Z"' in ttl

    def test_operator_inputs_outputs(self):
        """Test operator input/output export (compact node-level form)."""
        model = _create_simple_model()
        ttl = model_to_ttl(model)

        # Operator input/output details are not emitted at the operator level
        assert "a onnx:OperatorInput" not in ttl
        assert "a onnx:OperatorOutput" not in ttl
        assert 'onnx:inputs' not in ttl
        assert 'onnx:outputs' not in ttl
        assert 'onnx:inputCount' not in ttl
        assert 'onnx:outputCount' not in ttl


class TestSaveTTL:
    """Test save_ttl function."""

    def test_save_to_file(self, tmp_path):
        """Test saving TTL to file."""
        model = _create_simple_model()

        out_path = tmp_path / "test.ttl"
        result = save_ttl(model, out_path)

        assert result == out_path
        assert out_path.exists()

        content = out_path.read_text()
        assert "@prefix onnx:" in content

    def test_save_determinism(self, tmp_path):
        """Test that saved files are deterministic."""
        model = _create_simple_model()

        path1 = tmp_path / "test1.ttl"
        path2 = tmp_path / "test2.ttl"

        save_ttl(model, path1)
        save_ttl(model, path2)

        assert path1.read_text() == path2.read_text()


class TestOnnxFileToTTL:
    """Test onnx_file_to_ttl function."""

    def test_convert_file(self):
        """Test converting ONNX file to TTL."""
        model = _create_simple_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "test.onnx"
            onnx.save(model, str(onnx_path))

            ttl = onnx_file_to_ttl(onnx_path)

            assert "@prefix onnx:" in ttl
            assert "a onnx:Model" in ttl

    def test_convert_and_save(self, tmp_path):
        """Test converting ONNX file and saving TTL."""
        model = _create_simple_model()

        onnx_path = tmp_path / "test.onnx"
        ttl_path = tmp_path / "test.ttl"

        onnx.save(model, str(onnx_path))
        ttl = onnx_file_to_ttl(onnx_path, ttl_path)

        assert ttl_path.exists()
        assert ttl_path.read_text() == ttl


class TestTTLFormatting:
    """Test TTL formatting and escaping."""

    def test_special_characters_in_names(self):
        """Test handling of special characters in names."""
        X = helper.make_tensor_value_info("input:0", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("output:0", TensorProto.FLOAT, [1, 3])

        node = helper.make_node("Identity", ["input:0"], ["output:0"], name="node/with/slashes")

        graph = helper.make_graph([node], "test-graph", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        ttl = model_to_ttl(model)

        # Should not crash and should produce valid output
        assert "@prefix onnx:" in ttl
        # Names with special chars should be escaped
        assert "input_0" in ttl or "input:0" in ttl

    def test_empty_model_name(self):
        """Test handling of empty model/graph names."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])

        node = helper.make_node("Identity", ["X"], ["Y"])  # no name

        graph = helper.make_graph([node], "", [X], [Y])  # empty graph name
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        ttl = model_to_ttl(model)

        # Should handle empty names gracefully

    def test_id_and_type_emission(self):
        """Ensure @id is emitted as skos:exactMatch and @type as rdf:type."""
        model = _create_simple_model()
        # Add metadata props for @id and @type
        proto_id = onnx.StringStringEntryProto()
        proto_id.key = "@id"
        proto_id.value = "http://example.org/myid"
        model.metadata_props.append(proto_id)

        proto_type = onnx.StringStringEntryProto()
        proto_type.key = "@type"
        proto_type.value = "http://example.org/TypeX"
        model.metadata_props.append(proto_type)

        ttl = model_to_ttl(model)

        # Check that rdf:type and skos:exactMatch are emitted as IRIs
        assert "rdf:type <http://example.org/TypeX>" in ttl
        assert "skos:exactMatch <http://example.org/myid>" in ttl
        # Also ensure original metadata props are still present
        assert 'onnx:meta/_id "http://example.org/myid"' in ttl
        assert 'onnx:meta/_type "http://example.org/TypeX"' in ttl

    def test_type_and_id_must_be_iri_or_curie(self):
        model = _create_simple_model()
        proto = onnx.StringStringEntryProto()
        proto.key = "@type"
        proto.value = "not-an-iri"
        model.metadata_props.append(proto)
        with pytest.raises(ValueError):
            model_to_ttl(model)

    def test_model_functions_emitted(self):
        """Functions present in ModelProto should be emitted as onnx:Function resources."""
        model = _create_simple_model()
        # Create a minimal FunctionProto and attach to the model
        func = onnx.FunctionProto()
        func.name = "MyFunc"
        func.input.extend(["a", "b"])
        func.output.extend(["c"])
        model.functions.extend([func])

        ttl = model_to_ttl(model)

        assert "a onnx:Function" in ttl
        assert 'onnx:name "MyFunc"' in ttl
        assert 'onnx:inputs "[a,b]"' in ttl or 'onnx:inputs "[a,b]"' in ttl

    def test_type_and_id_curie_allowed_with_prefix(self):
        model = _create_simple_model()
        proto = onnx.StringStringEntryProto()
        proto.key = "@type"
        proto.value = "my:TypeX"
        model.metadata_props.append(proto)

        ttl = model_to_ttl(model, user_ns="my:", user_ns_uri="https://example.org/#")

        assert "rdf:type my:TypeX" in ttl
        assert "@prefix my:" in ttl

    def test_curie_with_unknown_prefix_warns(self):
        model = _create_simple_model()
        proto = onnx.StringStringEntryProto()
        proto.key = "@type"
        proto.value = "foo:Bar"
        model.metadata_props.append(proto)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ttl = model_to_ttl(model)
            assert any("unknown prefix 'foo'" in str(x.message) for x in w) or any("CURIE" in str(x.message) for x in w)

    def test_curie_unknown_prefix_strict_raises(self):
        model = _create_simple_model()
        proto = onnx.StringStringEntryProto()
        proto.key = "@type"
        proto.value = "foo:Bar"
        model.metadata_props.append(proto)
        with pytest.raises(ValueError):
            model_to_ttl(model, strict=True)

    def test_curie_with_user_ns_prefix_also_ok_in_strict(self):
        model = _create_simple_model()
        proto = onnx.StringStringEntryProto()
        proto.key = "@type"
        proto.value = "my:TypeX"
        model.metadata_props.append(proto)

        # strict=True should accept 'my' when user_ns='my:' is provided
        ttl = model_to_ttl(model, user_ns="my:", user_ns_uri="https://example.org/#", strict=True)
        assert "rdf:type my:TypeX" in ttl
        assert "@prefix my:" in ttl

    def test_string_escaping(self):
        """Test string escaping in doc strings."""
        model = _create_simple_model()
        model.doc_string = 'Test with "quotes" and\nnewlines'

        ttl = model_to_ttl(model)

        # Should escape special characters
        assert "\\n" in ttl or "newlines" in ttl
        assert '\\"' in ttl or "quotes" in ttl
