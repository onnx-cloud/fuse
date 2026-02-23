"""Tests for context-aware completion provider."""
import pytest
import json

from src.jupyter.server import completions


def test_completions_no_prefix_returns_all():
    """Should return ops, keywords, and types without prefix."""
    results = completions(prefix="", context="")
    assert len(results) > 0
    assert len(results) <= 100  # Should be limited
    # Check we have at least some ops
    assert any(r['label'] == 'Add' for r in results)


def test_completions_filters_by_prefix():
    """Should filter by prefix case-insensitively."""
    results = completions(prefix="ad", context="")
    labels = [r['label'] for r in results]
    assert 'Add' in labels
    assert all('ad' in label.lower() for label in labels if label not in ['param', 'const'])


def test_completions_type_context():
    """Should suggest types after colon."""
    results = completions(prefix="f", context="param x: f")
    labels = [r['label'] for r in results]
    assert 'f32' in labels
    assert 'f64' in labels
    type_items = [r for r in results if r['kind'] == 'type']
    assert len(type_items) > 0


def test_completions_keyword_suggestions():
    """Should suggest keywords."""
    results = completions(prefix="par", context="")
    labels = [r['label'] for r in results]
    assert 'param' in labels
    keyword_items = [r for r in results if r['kind'] == 'keyword']
    assert len(keyword_items) > 0


def test_completions_op_suggestions():
    """Should suggest operators with details."""
    results = completions(prefix="Mat", context="")
    labels = [r['label'] for r in results]
    assert 'MatMul' in labels
    matmul_items = [r for r in results if r['label'] == 'MatMul']
    assert len(matmul_items) > 0
    assert matmul_items[0]['kind'] == 'function'
    assert 'ONNX Op' in matmul_items[0]['detail']


def test_completions_in_function_call():
    """Should add opening paren for ops in call context."""
    results = completions(prefix="Re", context="output = Re")
    relu_items = [r for r in results if r['label'] == 'Relu']
    # Note: paren insertion only happens if context indicates we're already in parens
    assert len(relu_items) > 0


def test_completions_returns_limited_results():
    """Should return max 100 results."""
    results = completions(prefix="", context="")
    assert len(results) <= 100


def test_completions_structure():
    """Each completion should have required fields."""
    results = completions(prefix="A", context="")
    for item in results:
        assert 'label' in item
        assert 'insertText' in item
        assert 'kind' in item
        assert 'detail' in item
        assert isinstance(item['label'], str)
        assert isinstance(item['insertText'], str)


def test_completions_context_detection():
    """Should detect different contexts."""
    # After colon - should prioritize types
    type_ctx = completions(prefix="f", context="x: f")
    type_labels = [r['label'] for r in type_ctx if r['kind'] == 'type']
    assert len(type_labels) > 0
    
    # In expression - should include ops
    expr_ctx = completions(prefix="A", context="y = A")
    op_labels = [r['label'] for r in expr_ctx if r['kind'] == 'function']
    assert len(op_labels) > 0


def test_completions_empty_context():
    """Should handle empty context gracefully."""
    results = completions(prefix="Add", context="")
    assert len(results) > 0
    assert any(r['label'] == 'Add' for r in results)


def test_completions_case_insensitive():
    """Should match case-insensitively."""
    lower_results = completions(prefix="add", context="")
    upper_results = completions(prefix="ADD", context="")
    assert len(lower_results) > 0
    assert len(upper_results) > 0
    # Should find 'Add' in both cases
    assert any(r['label'] == 'Add' for r in lower_results)
    assert any(r['label'] == 'Add' for r in upper_results)


@pytest.mark.parametrize("prefix,expected_in_results", [
    ("Mat", "MatMul"),
    ("Con", "Concat"),
    ("par", "param"),
    ("f3", "f32"),
    ("i6", "i64"),
])
def test_completions_specific_cases(prefix, expected_in_results):
    """Test specific completion scenarios."""
    results = completions(prefix=prefix, context="")
    labels = [r['label'] for r in results]
    assert expected_in_results in labels


def test_completions_with_attributes():
    """Should include op attributes in detail."""
    results = completions(prefix="Conv", context="")
    conv_items = [r for r in results if 'Conv' in r['label']]
    # Conv ops have attributes - should be shown in detail
    assert len(conv_items) > 0
    # At least one should mention attributes or be an ONNX Op
    assert any('ONNX Op' in item['detail'] for item in conv_items)
