from src.sanitizer import sanitize_ast


def test_unused_param_warns():
    ast = [{"type": "fn", "name": "f", "params": [{"name": "a"}], "body": [{"return": 1}]}]
    r = sanitize_ast(ast)
    assert any("appears unused" in w["message"].lower() for w in r["warnings"])


def test_unused_import_warns():
    ast = [{"type": "import", "name": "lib", "alias": "lib"}, {"type": "fn", "name": "f", "params": [], "body": [{"return": 1}]}]
    r = sanitize_ast(ast)
    assert any("import 'lib' appears unused" in w["message"].lower() or "import 'lib' appears unused" in w["message"].lower() for w in r["warnings"])


def test_return_unknown_var_warns():
    ast = [{"type": "fn", "name": "f", "params": [], "body": [{"return": "x"}]}]
    r = sanitize_ast(ast)
    assert any("return value 'x'" in w["message"].lower() for w in r["warnings"])
