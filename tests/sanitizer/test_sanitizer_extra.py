from src.sanitizer import sanitize_ast


def test_duplicate_top_level_decls_error():
    ast = [
        {"type": "fn", "name": "foo"},
        {"type": "fn", "name": "foo"},
    ]
    r = sanitize_ast(ast)
    assert any("duplicate declaration" in e["message"].lower() for e in r["errors"])


def test_duplicate_param_in_function_errors():
    ast = [{"type": "fn", "name": "f", "params": [{"name": "a"}, {"name": "a"}]}]
    r = sanitize_ast(ast)
    assert any("duplicate parameter" in e["message"].lower() for e in r["errors"])


def test_import_without_source_warns():
    ast = [{"type": "import", "name": "lib"}]
    r = sanitize_ast(ast)
    assert any("has no 'source' or 'variants'" in w["message"].lower() for w in r["warnings"])


def test_fused_tensors_missing_file_warns():
    ast = [{"type": "const", "name": "big", "value": {"fused_tensors": {}}}]
    r = sanitize_ast(ast)
    assert any("missing a valid file" in w["message"].lower() for w in r["warnings"])


def test_multiple_fns_without_namespace_warns():
    ast = [{"type": "fn", "name": "a"}, {"type": "fn", "name": "b"}]
    r = sanitize_ast(ast)
    assert any("multiple top-level functions" in w["message"].lower() for w in r["warnings"])
