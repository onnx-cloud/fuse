import warnings
import onnx
from onnx import helper
from src.io.path_utils import artifact_path_for


def make_model_with_meta(meta: dict):
    # simple empty model with given metadata
    g = helper.make_graph([], "", [], [])
    m = helper.make_model(g)
    for k, v in meta.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=str(v)))
    return m


def test_artifact_path_accepts_module_alias(tmp_path):
    m = make_model_with_meta({"module": "foo.bar", "version": "1.0"})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        p = artifact_path_for(m, base=str(tmp_path))
    assert "foo" in p
    assert any("module" in str(warn.message) for warn in w)


def test_artifact_path_prefers_domain():
    m = make_model_with_meta({"module": "foo", "domain": "baz"})
    p = artifact_path_for(m, base="/tmp")
    assert "baz" in p and "foo" not in p
