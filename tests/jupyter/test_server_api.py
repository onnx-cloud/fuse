from src.jupyter.server import get_ops, completions, get_op_attributes, map_error


def test_get_ops_returns_list():
    ops = get_ops()
    assert isinstance(ops, list)


def test_completions_prefix():
    # best-effort: completions should accept a prefix and return a list
    c = completions(prefix="A")
    assert isinstance(c, list)


def test_op_attributes_structure():
    a = get_op_attributes("Add")
    assert isinstance(a, list)


def test_map_error_basic():
    m = map_error("boom")
    assert "message" in m and "stacktrace" in m and "friendly" in m
