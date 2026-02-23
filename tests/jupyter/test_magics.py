import pytest

IPY = pytest.importorskip("IPython")
from IPython import InteractiveShell  # noqa: E402

import onnx  # noqa: E402
from src.fuse_jupyter.ipython import load_ipython_extension  # noqa: E402


def _clear_ns(ip):
    for k in list(ip.user_ns.keys()):
        if k.startswith("_fuse_") or k == "_fuse_model":
            ip.user_ns.pop(k, None)


def test_cell_magic_parses_and_lowers(tmp_path):
    ip = InteractiveShell.instance()
    load_ipython_extension(ip)
    _clear_ns(ip)

    code = """
    node m(x: f32) -> f32 { return x }
    """
    # run cell magic
    _ = ip.run_cell_magic("fuse", "", code)
    assert "_fuse_model" in ip.user_ns
    m = ip.user_ns.get("_fuse_model")
    assert m is not None
    # Should be a valid ModelProto
    assert isinstance(m, onnx.ModelProto)

    # running again overwrites variable
    code2 = "node n(x: f32) -> f32 { return x }"
    ip.run_cell_magic("fuse", "", code2)
    assert "_fuse_model" in ip.user_ns


def test_cell_magic_handles_parse_errors(tmp_path, capsys):
    ip = InteractiveShell.instance()
    load_ipython_extension(ip)
    _clear_ns(ip)

    bad = "node bad(x: f32 -> f32 { return x }"  # malformed
    _ = ip.run_cell_magic("fuse", "", bad)
    # No model should be exposed
    assert "_fuse_model" not in ip.user_ns


def test_line_magic_export(tmp_path):
    ip = InteractiveShell.instance()
    load_ipython_extension(ip)
    _clear_ns(ip)

    code = "node m(x: f32) -> f32 { return x }"
    ip.run_cell_magic("fuse", "", code)
    out = tmp_path / "m.onnx"
    ip.run_line_magic("fuse", f"export {out}")
    assert out.exists()
