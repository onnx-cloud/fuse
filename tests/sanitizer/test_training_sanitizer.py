from src.sanitizer import sanitize_ast


def test_sanitizer_emits_structured_training_warnings():
    ast = [
        {"type": "meta", "name": "fuse.training", "value": {"optimizer": "Adam"}},
        {"type": "param", "name": "W", "trainable": True},
    ]

    res = sanitize_ast(ast)
    warnings = res.get("warnings", [])
    # Expect at least one structured TRAIN.MISSING_STATE warning for W.m or W.v
    assert any(w.get("code") == "TRAIN.MISSING_STATE" and w.get("param") == "W" for w in warnings)
