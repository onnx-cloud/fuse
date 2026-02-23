from src.sanitizer import sanitize_ast


def test_noncanonical_op_name_accepted_with_onnx_fallback():
    # 'matmul' lowercase should be accepted via ONNX schema fallback and
    # produce a non-fatal warning about non-canonical case, not an error.
    ast = [{"type": "fn", "call": "matmul"}]
    res = sanitize_ast(ast)
    warnings = res.get("warnings", [])
    errors = res.get("errors", [])

    assert not errors, f"Unexpected errors: {errors}"
    assert any("non-canonical case" in w.get("message", "").lower() or "matmul" in w.get("message", "").lower() for w in warnings)


def test_training_meta_default_optimizer_warns():
    # An empty @training {} should warn about defaulting to 'adam'
    ast = [{"type": "meta", "name": "fuse.training", "value": {}}]
    res = sanitize_ast(ast)
    warnings = res.get("warnings", [])

    assert any("defaulting to 'adam'" in w.get("message", "").lower() for w in warnings), f"Expected default optimizer warning, got: {warnings}"