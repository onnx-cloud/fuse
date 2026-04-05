import warnings
import onnx
import pytest
from onnx import helper
from src.io.path_utils import artifact_path_for


def make_model_with_meta(meta: dict):
    # simple empty model with given metadata
    g = helper.make_graph([], "", [], [])
    m = helper.make_model(g)
    for k, v in meta.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=str(v)))
    return m


def test_artifact_path_requires_domain(tmp_path):
    # The module alias is no longer supported; only 'domain' key is accepted
    from onnx import helper
    g = helper.make_graph([], "gname", [], [])
    m = helper.make_model(g)
    m.metadata_props.append(onnx.StringStringEntryProto(key="module", value="foo.bar"))
    m.metadata_props.append(onnx.StringStringEntryProto(key="version", value="1.0"))
    # Should raise because module key is no longer supported
    with pytest.raises(ValueError, match="domain"):
        artifact_path_for(m, base=str(tmp_path))


def test_artifact_path_prefers_domain():
    from onnx import helper
    g = helper.make_graph([], "gn", [], [])
    m = helper.make_model(g)
    m.metadata_props.append(onnx.StringStringEntryProto(key="module", value="foo"))
    m.metadata_props.append(onnx.StringStringEntryProto(key="domain", value="baz"))
    p = artifact_path_for(m, base="/tmp")
    assert "baz" in p and "foo" not in p


def test_graph_name_overrides_title(tmp_path):
    # create a model whose graph has lowercase name but metadata title is
    # capitalized; artifact path should use the graph name exactly
    from onnx import helper
    g = helper.make_graph([], "mygraph", [], [])
    m = helper.make_model(g)
    # add metadata title which would previously win
    m.metadata_props.append(onnx.StringStringEntryProto(key="title", value="MyGraph"))
    m.metadata_props.append(onnx.StringStringEntryProto(key="domain", value="dom"))
    p = artifact_path_for(m, base=str(tmp_path))
    assert p.endswith("/dom/unversioned/mygraph.onnx"), f"got {p}"


def test_missing_graph_name_errors(tmp_path):
    # model without graph name should raise ValueError
    from onnx import helper
    g = helper.make_graph([], "", [], [])
    m = helper.make_model(g)
    m.metadata_props.append(onnx.StringStringEntryProto(key="domain", value="dom"))
    with pytest.raises(ValueError):
        artifact_path_for(m, base=str(tmp_path))


