import io
import sys

from src.cli import main


def _capture_stdout(func, *args, **kwargs):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        rc = func(*args, **kwargs)
        out = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    return rc, out


def test_decompile_and_audit_alias_invoke_cmd_decompile(monkeypatch):
    called = {}

    def fake_cmd_decompile(files, **kwargs):
        called['args'] = (files, kwargs)
        return [(files[0], ['out1'], None)]

    monkeypatch.setattr('src.cli.cli_commands.cmd_decompile', fake_cmd_decompile)
    # ensure file expansion returns our provided names (avoid filesystem dependency)
    monkeypatch.setattr('src.cli.cli_helpers.find_fuse_files', lambda v: list(v) if isinstance(v, (list, tuple)) else [v])

    rc, out = _capture_stdout(main, ['decompile', '-f', 'm.onnx', '--proto'])
    assert rc == 0
    assert called['args'][0] == ['m.onnx']
    assert called['args'][1].get('proto') is True

    rc2, out2 = _capture_stdout(main, ['audit', '-f', 'm.onnx', '--proto'])
    assert rc2 == 0
    assert called['args'][0] == ['m.onnx']
    assert called['args'][1].get('proto') is True
