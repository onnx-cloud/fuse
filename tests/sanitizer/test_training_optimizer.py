from src.sanitizer import sanitize_ast


def test_training_default_optimizer_warns():
    ast = [{"type": "meta", "name": "fuse.training", "value": {}}]
    r = sanitize_ast(ast)
    assert any("defaulting to 'adam'" in w["message"].lower() for w in r["warnings"])


def test_training_builtin_optimizer_ok():
    ast = [{"type": "meta", "name": "fuse.training", "value": {"optimizer": "SGD"}}]
    r = sanitize_ast(ast)
    assert not r["errors"]
    assert not any("not a known builtin" in w["message"].lower() for w in r["warnings"])


def test_training_optimizer_refers_to_node_ok():
    ast = [
        {"type": "meta", "name": "fuse.training", "value": {"optimizer": "MyOpt"}},
        {"type": "fn", "name": "MyOpt"},
    ]
    r = sanitize_ast(ast)
    assert not any("not a known builtin" in w["message"].lower() for w in r["warnings"])


def test_training_optimizer_unresolved_warns():
    ast = [{"type": "meta", "name": "fuse.training", "value": {"optimizer": "MysteryOpt"}}]
    r = sanitize_ast(ast)
    assert any("not a known builtin" in w["message"].lower() for w in r["warnings"])
