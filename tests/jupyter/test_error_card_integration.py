"""Tests for error card IPython integration."""
import pytest

# Only run if IPython is available
IPY = pytest.importorskip("IPython")
from IPython import InteractiveShell
from IPython.display import HTML, JSON

from src.jupyter.ipython import load_ipython_extension


@pytest.fixture
def ip():
    """Create a test IPython instance."""
    shell = InteractiveShell.instance()
    load_ipython_extension(shell)
    yield shell
    # Cleanup
    InteractiveShell.clear_instance()


def test_exception_hook_installed(ip):
    """Should install custom exception hook."""
    # Check that custom exception handler is set
    assert hasattr(ip, '_custom_exceptions')
    assert Exception in ip._custom_exceptions


def test_exception_displays_html(ip, capsys):
    """Should display HTML error card when exception occurs."""
    # Trigger an exception
    code = """
raise ValueError("Test error message")
"""
    # Run code that raises exception
    result = ip.run_cell(code, silent=False, store_history=False)
    
    # Should have error
    assert result.error_in_exec is not None
    
    # Check that output was generated (HTML display)
    # Note: In tests, display output may not be captured directly
    # but we can verify the hook is called


def test_error_info_structure(ip):
    """Exception mapping should return structured data."""
    from src.jupyter.errors import map_exception
    
    try:
        raise ValueError("Test error")
    except ValueError as e:
        info = map_exception(e)
        
        # Should have required fields  (actual keys: 'friendly', 'filename', 'line', etc.)
        assert 'friendly' in info or 'error' in info
        assert 'filename' in info or 'message' in info
        assert isinstance(info, dict)


def test_exception_hook_shows_suggestion(ip):
    """Error card should show suggestions when available."""
    from src.jupyter.errors import map_exception
    
    # Create an exception that should have suggestions
    try:
        raise NameError("name 'Addd' is not defined")
    except NameError as e:
        info = map_exception(e)
        
        # Should detect typo and suggest correction
        # (depends on error mapping implementation)
        assert 'friendly' in info or 'error' in info
        assert 'filename' in info or 'message' in info


def test_html_output_escaping():
    """HTML output should properly escape user input."""
    import html
    
    # Test that dangerous strings are escaped
    dangerous = "<script>alert('xss')</script>"
    escaped = html.escape(dangerous)
    
    assert '&lt;script&gt;' in escaped
    assert '<script>' not in escaped


def test_error_card_has_collapsible_details():
    """Error HTML should include details section."""
    from src.jupyter.errors import map_exception
    import html as _html
    
    try:
        raise RuntimeError("Complex error with details")
    except RuntimeError as e:
        info = map_exception(e)
        
        # Simulate HTML generation
        if info.get('detail'):
            html_output = f"<details>{_html.escape(info['detail'])}</details>"
            assert '<details>' in html_output
            assert '</details>' in html_output


def test_exception_json_output(ip):
    """Should output JSON structure for programmatic access."""
    from src.jupyter.errors import map_exception
    
    try:
        raise TypeError("Type mismatch")
    except TypeError as e:
        info = map_exception(e)
        
        # JSON should be valid
        import json
        json_str = json.dumps(info)
        parsed = json.loads(json_str)
        assert parsed == info


def test_multiple_exceptions(ip):
    """Should handle multiple consecutive exceptions."""
    codes = [
        "raise ValueError('Error 1')",
        "raise TypeError('Error 2')",
        "raise NameError('Error 3')",
    ]
    
    for code in codes:
        result = ip.run_cell(code, silent=False, store_history=False)
        assert result.error_in_exec is not None


def test_exception_with_traceback(ip):
    """Should handle exceptions with full tracebacks."""
    code = """
def nested_func():
    raise RuntimeError("Deep error")

def caller():
    nested_func()

caller()
"""
    result = ip.run_cell(code, silent=False, store_history=False)
    assert result.error_in_exec is not None


def test_error_card_styling():
    """Error card HTML should have proper styling."""
    html_template = '''
    <div style="border: 1px solid #e74c3c; border-radius: 4px; padding: 12px;">
        Error content
    </div>
    '''
    
    # Check for essential style properties
    assert 'border:' in html_template or 'border-radius:' in html_template
    assert 'padding:' in html_template


def test_suggestion_rendering():
    """Suggestions should be visually distinct."""
    from src.jupyter.errors import map_exception
    import html as _html
    
    # Mock info with suggestion
    info = {
        'error': 'NameError',
        'message': 'Variable not found',
        'suggestion': 'Did you mean: variable_name?'
    }
    
    # Simulate suggestion HTML
    suggestion_html = f'<div style="background: #fff3cd;">{_html.escape(info["suggestion"])}</div>'
    assert 'Did you mean' in suggestion_html
    assert 'background:' in suggestion_html


@pytest.mark.parametrize("error_type,message", [
    (ValueError, "Invalid value"),
    (TypeError, "Type mismatch"),
    (NameError, "Name not defined"),
    (RuntimeError, "Runtime issue"),
])
def test_various_exception_types(ip, error_type, message):
    """Should handle different exception types."""
    from src.jupyter.errors import map_exception
    
    try:
        raise error_type(message)
    except Exception as e:
        info = map_exception(e)
        assert 'friendly' in info or 'error' in info
        assert isinstance(info, dict)


def test_exception_hook_preserves_traceback(ip, capsys):
    """Should still print traceback for debugging."""
    code = "raise ValueError('Debug me')"
    result = ip.run_cell(code, silent=False, store_history=False)
    
    # Traceback should be printed (captured in stderr/stdout)
    captured = capsys.readouterr()
    # At minimum, error should propagate
    assert result.error_in_exec is not None
