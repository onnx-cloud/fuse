from src.jupyter.server import kernel_symbols


def test_kernel_symbols_no_app():
    res = kernel_symbols('nonexistent')
    assert res["ok"] is False and "error" in res
