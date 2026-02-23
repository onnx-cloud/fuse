def test_import_src_package():
    # Quick smoke test to ensure the package is importable
    import importlib
    spec = importlib.util.find_spec('src')
    assert spec is not None, "src package should be importable"
    import src
    assert hasattr(src, '__version__'), "src package should expose __version__"
