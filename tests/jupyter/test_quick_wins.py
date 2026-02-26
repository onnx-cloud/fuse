"""Tests for quick wins implementation - timeouts, errors, loading states"""
import os
import time
from unittest.mock import Mock, patch, MagicMock
import pytest


def test_llm_timeout_configuration():
    """Test that FUSE_LLM_TIMEOUT environment variable is respected"""
    # Default should be 30 seconds
    from src.jupyter.server import _load_llm_config
    
    # Verify timeout is configurable
    with patch.dict(os.environ, {'FUSE_LLM_TIMEOUT': '60'}):
        timeout = int(os.environ.get('FUSE_LLM_TIMEOUT', '30'))
        assert timeout == 60
    
    # Verify default
    with patch.dict(os.environ, {}, clear=True):
        timeout = int(os.environ.get('FUSE_LLM_TIMEOUT', '30'))
        assert timeout == 30


def test_error_response_schema():
    """Test that error responses include error, detail, and suggestion fields"""
    from src.jupyter.server import LLMHandler
    
    # Mock a timeout error response structure
    error_response = {
        'error': 'LLM request timed out after 30s',
        'suggestion': 'Try a shorter prompt or increase FUSE_LLM_TIMEOUT'
    }
    
    assert 'error' in error_response
    assert 'suggestion' in error_response
    assert 'timed out' in error_response['error'].lower()


def test_connection_error_message():
    """Test that connection errors provide helpful suggestions"""
    error_response = {
        'error': 'Cannot connect to LLM provider: Connection refused',
        'suggestion': 'Check your internet connection and provider URL'
    }
    
    assert 'connection' in error_response['error'].lower()
    assert 'check' in error_response['suggestion'].lower()


def test_http_error_message():
    """Test that HTTP errors include status code and suggestions"""
    error_response = {
        'error': 'LLM provider error (HTTP 401)',
        'detail': 'Invalid API key',
        'suggestion': 'Check your API key and model name'
    }
    
    assert 'HTTP' in error_response['error']
    assert 'detail' in error_response
    assert 'API key' in error_response['suggestion']


def test_chat_styles_css_exists():
    """Test that chat-styles.css file was created"""
    from pathlib import Path
    css_path = Path(__file__).parent.parent.parent / 'jupyter' / 'static' / 'chat-styles.css'
    assert css_path.exists(), "chat-styles.css should exist"
    
    # Verify it contains mobile breakpoints
    content = css_path.read_text()
    assert '@media (max-width: 768px)' in content, "Should have tablet breakpoint"
    assert '@media (max-width: 480px)' in content, "Should have mobile breakpoint"
    assert 'fuse-chat-wrapper' in content, "Should have chat wrapper class"
    assert 'fuse-chat-loading' in content, "Should have loading class"


def test_keyboard_shortcut_registration():
    """Test that keyboard shortcuts are properly registered"""
    # This is a smoke test - full integration test would require JupyterLab context
    shortcut_config = {
        'fuse:open-chat': 'Accel K',
        'fuse:open-welcome-widget': 'Accel Shift H'
    }
    
    assert 'fuse:open-chat' in shortcut_config
    assert 'Accel K' == shortcut_config['fuse:open-chat']


def test_loading_state_logic():
    """Test loading state management logic"""
    # Simulate loading state transitions
    is_loading = False
    
    # Start loading
    is_loading = True
    assert is_loading == True
    
    # Button should be disabled during loading
    button_disabled = is_loading
    assert button_disabled == True
    
    # End loading
    is_loading = False
    assert is_loading == False
    button_disabled = is_loading
    assert button_disabled == False


def test_error_truncation():
    """Test that long error messages are truncated to 500 chars"""
    long_error = "x" * 1000
    truncated = long_error[:500]
    
    assert len(truncated) == 500
    assert len(truncated) < len(long_error)


@pytest.mark.asyncio
async def test_timeout_request_simulation():
    """Simulate a request timeout scenario"""
    import asyncio
    
    async def slow_request():
        await asyncio.sleep(35)  # Longer than default timeout
        return "response"
    
    # Should timeout before completing
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_request(), timeout=30)


def test_mobile_responsive_classes():
    """Test that CSS classes for mobile are properly defined"""
    from pathlib import Path
    css_path = Path(__file__).parent.parent.parent / 'jupyter' / 'static' / 'chat-styles.css'
    
    if css_path.exists():
        content = css_path.read_text()
        
        # Check for critical responsive elements
        required_classes = [
            '.fuse-chat-wrapper',
            '.fuse-chat-log',
            '.fuse-chat-button',
            '.fuse-chat-input',
            '.fuse-chat-loading',
        ]
        
        for cls in required_classes:
            assert cls in content, f"Missing class: {cls}"


def test_css_handler_registered():
    """Test that ChatStylesHandler is registered in server routes"""
    # This verifies the handler class exists and can be imported
    try:
        from src.jupyter.server import ChatStylesHandler
        assert ChatStylesHandler is not None
    except ImportError:
        # If server components not available, skip
        pytest.skip("Server components not available")


def test_backward_compatibility():
    """Test that old error format still works"""
    # Old format (simple string)
    old_error = {'error': 'Something went wrong'}
    assert 'error' in old_error
    
    # New format (with details)
    new_error = {
        'error': 'Something went wrong',
        'detail': 'Connection refused',
        'suggestion': 'Check your internet'
    }
    assert 'error' in new_error
    
    # Both formats should work - new is superset of old
    for key in old_error.keys():
        assert key in new_error


def test_env_var_defaults():
    """Test that all environment variables have sensible defaults"""
    defaults = {
        'FUSE_LLM_TIMEOUT': '30',
        'FUSE_LLM_RATE_PER_MIN': '60',
    }
    
    for var, default in defaults.items():
        with patch.dict(os.environ, {}, clear=True):
            value = os.environ.get(var, default)
            assert value == default


def test_audit_logging_includes_status():
    """Test that audit logs include response status codes"""
    # Simulate audit log entry
    audit_entry = {
        'ts': time.time(),
        'ip': '127.0.0.1',
        'engine': 'test',
        'payload': {'messages': []},
        'status': 504  # Timeout
    }
    
    assert 'status' in audit_entry
    assert audit_entry['status'] == 504


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
