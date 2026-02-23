from src.sanitizer import sanitize_ast


def test_conflicting_train_and_frozen():
    ast = [
        {"type": "param", "name": "W", "trainable": True},
        {"type": "param", "name": "W", "trainable": False},
    ]
    r = sanitize_ast(ast)
    assert any("conflicting" in e["message"].lower() for e in r["errors"])


def test_train_without_training_meta_warns():
    ast = [{"type": "param", "name": "W", "trainable": True}]
    r = sanitize_ast(ast)
    assert any("@train used" in w["message"] for w in r["warnings"])


def test_lowercase_op_name_warns_and_is_accepted():
    ast = [{"type": "fn", "call": "matmul"}]
    r = sanitize_ast(ast)
    assert any("non-canonical case" in w["message"].lower() for w in r["warnings"])
    assert not r["errors"]


def test_training_meta_without_train_warns():
    ast = [{"type": "meta", "name": "fuse.training", "value": {"optimizer": "adam"}}]
    r = sanitize_ast(ast)
    assert any("@training metadata present" in w["message"] for w in r["warnings"])


def test_unknown_op_errors():
    ast = [{"type": "node", "call": "i_am_not_an_op"}]
    r = sanitize_ast(ast)
    assert any("unknown operator" in e["message"].lower() for e in r["errors"])
