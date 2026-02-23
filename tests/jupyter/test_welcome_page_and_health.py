from pathlib import Path
from src.jupyter.server import map_error, get_ops, kernel_symbols


def test_welcome_static_exists():
    p = Path(__file__).resolve().parents[2] / "jupyter" / "static" / "welcome.html"
    assert p.exists(), f"welcome.html not found at {p}"


def test_health_endpoint_logic():
    # Call the health logic indirectly via imports
    # We can't start the server here, but HealthHandler logic checks imports
    # Sanity check map_error integration
    m = map_error('boom')
    assert 'message' in m and 'friendly' in m


def test_kernel_symbols_fallback():
    res = kernel_symbols('none')
    assert res.get('ok') is False and 'error' in res
