from src.sanitizer import sanitize_ast


def test_unused_const_warns():
    ast = [{"type": "const", "name": "C", "value": 1}, {"type": "fn", "name": "f", "params": [], "body": [{"return": 2}]}]
    r = sanitize_ast(ast)
    assert any("const 'c' appears unused" in w["message"].lower() for w in r["warnings"])


def test_type_alias_invalid_dim_warns():
    ast = [{"type": "type_alias", "name": "T", "type_decl": {"scalar": "f32", "dims": ["N", {"weird": True}]}}]
    r = sanitize_ast(ast)
    assert any("type alias 't' has invalid dimension" in w["message"].lower() for w in r["warnings"])


def test_type_alias_empty_scalar_warns():
    ast = [{"type": "type_alias", "name": "U", "type_decl": ""}]
    r = sanitize_ast(ast)
    assert any("type alias 'u' has empty scalar type" in w["message"].lower() for w in r["warnings"])
