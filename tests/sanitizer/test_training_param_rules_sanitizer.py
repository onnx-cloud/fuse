from src.sanitizer import sanitize_ast


def test_sanitizer_emits_param_state_expectation_for_conv_like():
    ast = [
        {"type": "meta", "name": "fuse.training", "value": {"optimizer": "adam"}},
        {"type": "param", "name": "conv.W", "trainable": True, "type_decl": {"scalar": "f32", "dims": [16, 3, 3]}},

    ]
    res = sanitize_ast(ast)
    warnings = res.get("warnings", [])
    assert any(w.get("code") == "TRAIN.PARAM_STATE_EXPECTATION" and w.get("param") == "conv.W" for w in warnings)


def test_sanitizer_does_not_emit_when_no_rule_matches():
    ast = [
        {"type": "meta", "name": "fuse.training", "value": {"optimizer": "adam"}},
        {"type": "param", "name": "other", "trainable": True, "type_decl": {"scalar": "f32", "dims": [10]}},
    ]
    res = sanitize_ast(ast)
    warnings = res.get("warnings", [])
    assert not any(w.get("code") == "TRAIN.PARAM_STATE_EXPECTATION" and w.get("param") == "other" for w in warnings)
