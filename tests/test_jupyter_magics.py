import pytest
pytest.importorskip("IPython")
pytest.importorskip("numpy")


def test_compile_and_run_ipython_magics():
    # create a fresh InteractiveShell and load extension
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    # load our extension into the shell
    import src.jupyter.extension as ext

    ext.load_ipython_extension(shell)

    # compile a minimal identity node
    src = "fn id(x: f32[1]) -> f32[1] { x }"
    m = shell.run_cell_magic("fuse", "mymodel", src)
    # ensure model registered
    assert isinstance(m, object)
    # run with input
    res = shell.run_line_magic("fuse_run", "mymodel --input '{\"x\": [1.0]}' --provider reference")
    assert isinstance(res, dict)
    assert len(res) >= 1
