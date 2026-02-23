import traceback
import pytest
pytest.importorskip("IPython")
from src.jupyter.session import SessionManager
from src.jupyter.display import mime_bundle
from src.jupyter.introspection import list_ops, list_symbols
from src.jupyter.errors import map_exception


def test_session_manager_basic():
    s = SessionManager()
    s.set_var('x', 1)
    assert s.get_var('x') == 1
    s.record_module('onnx')
    assert 'onnx' in s.modules


def test_mime_bundle_structure():
    b = mime_bundle(text='ok', html='<b>ok</b>', json_obj={'a': 1})
    assert 'text/plain' in b and 'text/html' in b and 'application/json' in b


def test_introspection_ops_list():
    ops = list_ops()
    # best-effort: may be empty, but should return a list
    assert isinstance(ops, list)


def test_map_exception_contains_fields():
    try:
        raise RuntimeError('boom')
    except Exception as e:
        info = map_exception(e)
        assert 'message' in info and 'stacktrace' in info and 'friendly' in info
