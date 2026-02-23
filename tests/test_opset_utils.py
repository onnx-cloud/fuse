from src.util.opset_utils import compute_opset_imports, SAFE_MAX_OPSET


def test_basic_core_and_extras_order():
    extra = {"b.com": 2, "a.com": 3}
    tuples = compute_opset_imports(18, extra)
    assert tuples[0] == ("", 18)
    # extras sorted alphabetically
    assert tuples[1] == ("a.com", 3)
    assert tuples[2] == ("b.com", 2)


def test_caps_to_safe_max():
    extra = {"x": 999}
    tuples = compute_opset_imports(999, extra)
    assert tuples[0] == ("", SAFE_MAX_OPSET)
    assert tuples[1] == ("x", SAFE_MAX_OPSET)


def test_accepts_string_and_int_inputs():
    extra = {"d": "7"}
    t1 = compute_opset_imports("9", extra)
    t2 = compute_opset_imports(9, extra)
    assert t1 == t2
    assert t1[1] == ("d", 7)
